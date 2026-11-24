import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features

try:
    from scripts.train_utils import (
        MetricsLogger,
        VisualAugmentor,
        build_warmup_cosine_scheduler,
        count_parameters,
        evaluate_heldout_mae,
        save_artifacts,
        set_seed,
        split_train_val_episodes,
    )
except ImportError:
    from train_utils import (
        MetricsLogger,
        VisualAugmentor,
        build_warmup_cosine_scheduler,
        count_parameters,
        evaluate_heldout_mae,
        save_artifacts,
        set_seed,
        split_train_val_episodes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 ACT 策略（支持 Warmup+Cosine、AMP、梯度累积、视觉增强与无泄露验证）"
    )
    parser.add_argument("--repo-id", type=str, default="omy_pnp", help="数据集仓库标识名")
    parser.add_argument("--dataset-root", type=str, default="./dataset/ACT_dataset", help="本地数据集根目录")
    parser.add_argument("--output-dir", type=str, default="./ckpt/act_y", help="模型与日志保存目录")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="可选：从已有本地 ACT checkpoint 继续微调",
    )

    parser.add_argument("--chunk-size", type=int, default=10, help="每次预测的动作块长度")
    parser.add_argument("--n-action-steps", type=int, default=10, help="推理时每次连续执行的动作步数")
    parser.add_argument("--steps", type=int, default=30000, help="总优化器更新步数")
    parser.add_argument("--batch-size", type=int, default=32, help="单次前向微批次大小")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="梯度累积步数（等效 batch = batch_size * grad_accum_steps）")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 ACTConfig 默认学习率")
    parser.add_argument("--warmup-steps", type=int, default=500, help="线性 Warmup 步数")
    parser.add_argument("--min-lr-ratio", type=float, default=0.05, help="余弦退火最低学习率比例")
    parser.add_argument("--grad-clip-norm", type=float, default=10.0, help="梯度裁剪最大范数（<=0 关闭）")
    parser.add_argument("--amp", action="store_true", help="在 CUDA 上开启 FP16 自动混合精度加速训练")

    parser.add_argument("--save-freq", type=int, default=5000, help="定期保存中间 checkpoint 间隔步数（0 关闭）")
    parser.add_argument("--log-freq", type=int, default=50, help="日志打印与指标落盘间隔步数")
    parser.add_argument("--target-loss", type=float, default=None, help="可选：EMA loss <= 该阈值时提前停止（默认关闭）")
    parser.add_argument("--val-episodes", type=int, default=4, help="留出最后 N 条轨迹作为无泄露开环验证集")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数（Windows 建议保持 0）")

    # 视觉数据增强（提升抗红牛罐干扰与光照鲁棒性）
    parser.add_argument(
        "--image-aug",
        action="store_true",
        help="一键开启推荐视觉增强（color_jitter=0.2, noise_std=0.02, erasing_prob=0.15）",
    )
    parser.add_argument("--aug-color-jitter", type=float, default=0.0, help="主视角亮度/对比度/饱和度抖动幅度")
    parser.add_argument("--aug-noise-std", type=float, default=0.02, help="主视角高斯噪声标准差（0 关闭）")
    parser.add_argument("--aug-erasing-prob", type=float, default=0.0, help="随机遮挡擦除概率（模拟局部干扰）")
    parser.add_argument("--aug-wrist", action="store_true", help="同时对腕部相机施加 0.5x 轻度增强")

    # 训练后离线评估开关
    parser.add_argument("--eval", dest="enable_eval", action="store_true", default=True, help="开启训练后开环验证")
    parser.add_argument("--no-eval", dest="enable_eval", action="store_false", help="关闭训练后开环验证")
    parser.add_argument("--eval-episode", type=int, default=0, help="当 val-episodes=0 时指定评估的轨迹编号")
    parser.add_argument("--eval-show-plot", action="store_true", default=False, help="是否弹窗展示评估曲线")
    parser.add_argument("--no-plot", dest="eval_show_plot", action="store_false", help="关闭评估曲线弹窗（默认已关闭）")
    parser.add_argument("--save-eval-plot", dest="eval_save_plot", action="store_true", default=True, help="保存评估对比图")
    parser.add_argument("--no-save-eval-plot", dest="eval_save_plot", action="store_false", help="不保存评估对比图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取数据集元信息与特征
    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 2. 初始化 ACT 配置与模型
    if args.resume_from is not None:
        cfg = ACTConfig.from_pretrained(args.resume_from)
        cfg.input_features = input_features
        cfg.output_features = output_features
        cfg.chunk_size = args.chunk_size
        cfg.n_action_steps = args.n_action_steps
        cfg.device = device.type
        policy = ACTPolicy.from_pretrained(args.resume_from, config=cfg)
    else:
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            device=device.type,
        )
        policy = ACTPolicy(cfg)

    policy.train()
    policy.to(device)
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=dataset_metadata.stats,
    )

    # 3. 加载数据集并按 episode 划分训练/验证集
    full_dataset = LeRobotDataset(
        args.repo_id, delta_timestamps=delta_timestamps, root=args.dataset_root
    )
    train_dataset, val_ep_ids = split_train_val_episodes(full_dataset, args.val_episodes)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    cj = max(args.aug_color_jitter, 0.2) if args.image_aug else args.aug_color_jitter
    ns = max(args.aug_noise_std, 0.02) if args.image_aug else args.aug_noise_std
    ep_prob = max(args.aug_erasing_prob, 0.15) if args.image_aug else args.aug_erasing_prob
    augmentor = VisualAugmentor(
        color_jitter=cj,
        noise_std=ns,
        erasing_prob=ep_prob,
        augment_wrist=args.aug_wrist,
    )

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else cfg.optimizer_lr
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=cfg.optimizer_weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    amp_enabled = args.amp and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    metrics_logger = MetricsLogger(save_dir)

    n_train_params, n_total_params = count_parameters(policy)
    eff_batch = args.batch_size * args.grad_accum_steps
    print(
        f"[Train ACT] device={device.type}, amp={amp_enabled}, "
        f"params={n_train_params / 1e6:.2f}M/{n_total_params / 1e6:.2f}M, "
        f"train_eps={dataset_metadata.total_episodes - len(val_ep_ids)}/{dataset_metadata.total_episodes}, "
        f"val_eps={val_ep_ids}, steps={args.steps}, batch={args.batch_size}x{args.grad_accum_steps}={eff_batch}, "
        f"lr={lr}, aug=(cj={cj}, noise={ns}, erase={ep_prob})"
    )

    # 4. 训练循环
    data_iter = iter(dataloader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            return next(data_iter)

    ema_loss: Optional[float] = None
    ema_alpha = 0.05
    best_ema_loss = float("inf")
    best_check_freq = min(250, max(args.log_freq, args.steps // 5, 1))
    warmup_gate = min(args.warmup_steps, max(1, args.steps // 2))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum_steps):
            batch = next_batch()
            batch = augmentor(batch)
            inp_batch = preprocessor(batch)
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                raw_loss, _ = policy.forward(inp_batch)
                scaled_loss = raw_loss / args.grad_accum_steps

            grad_scaler.scale(scaled_loss).backward()
            accum_loss += float(raw_loss.detach().item()) / args.grad_accum_steps

        grad_scaler.unscale_(optimizer)
        if args.grad_clip_norm > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=args.grad_clip_norm
                ).item()
            )
        else:
            grad_norm = 0.0

        prev_scale = grad_scaler.get_scale() if grad_scaler.is_enabled() else None
        grad_scaler.step(optimizer)
        grad_scaler.update()
        if prev_scale is None or grad_scaler.get_scale() >= prev_scale:
            scheduler.step()

        step_ms = (time.perf_counter() - t0) * 1000.0
        ema_loss = (
            accum_loss
            if ema_loss is None
            else (1.0 - ema_alpha) * ema_loss + ema_alpha * accum_loss
        )
        cur_lr = float(scheduler.get_last_lr()[0])
        vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            if device.type == "cuda"
            else 0.0
        )

        if step == 1 or step % args.log_freq == 0 or step == args.steps:
            metrics_logger.log(
                step=step,
                loss=accum_loss,
                ema_loss=ema_loss,
                lr=cur_lr,
                grad_norm=grad_norm,
                step_ms=step_ms,
                vram_mb=vram_mb,
            )
            print(
                f"step: {step:6d}/{args.steps} | loss: {accum_loss:.4f} | "
                f"ema: {ema_loss:.4f} | lr: {cur_lr:.2e} | "
                f"grad: {grad_norm:.2f} | {step_ms:.1f}ms | vram: {vram_mb:.0f}MB"
            )

        if step >= warmup_gate and (step % best_check_freq == 0 or step == args.steps):
            if ema_loss < best_ema_loss:
                best_ema_loss = ema_loss
                save_artifacts(save_dir / "best_ema", policy, preprocessor, postprocessor)

        if args.save_freq > 0 and step % args.save_freq == 0 and step < args.steps:
            ckpt_dir = save_dir / f"step_{step}"
            save_artifacts(ckpt_dir, policy, preprocessor, postprocessor)
            print(f"[Save] Saved intermediate checkpoint to: {ckpt_dir}")

        if args.target_loss is not None and step >= warmup_gate and ema_loss <= args.target_loss:
            print(
                f"[EarlyStop] step={step} ema_loss={ema_loss:.4f} <= {args.target_loss}，停止训练并保存模型"
            )
            break

    # 5. 保存最终模型权重、处理器与训练曲线
    save_artifacts(save_dir, policy, preprocessor, postprocessor)
    metrics_logger.save_plot(save_dir / "loss_curve.png")
    print(f"[Save] Saved final ACT policy & loss curve to: {save_dir}")

    # 6. 在留出验证集上执行离线开环评估
    if args.enable_eval:
        eval_eps = val_ep_ids if val_ep_ids else [args.eval_episode]
        plot_name = (
            f"eval_val_ep_{eval_eps[0]}.png"
            if val_ep_ids
            else f"eval_episode_{eval_eps[0]}.png"
        )
        plot_path = save_dir / plot_name if args.eval_save_plot else None
        evaluate_heldout_mae(
            policy=policy,
            dataset=full_dataset,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            episode_indices=eval_eps,
            save_plot_path=plot_path,
            show_plot=args.eval_show_plot,
        )
    else:
        print("[Eval] 模型评估已关闭（可使用 --eval 参数开启）。")


if __name__ == "__main__":
    main()
