"""
SmolVLA 策略微调脚本（支持一阶段只训动作专家 & 二阶段解冻 VLM 联合精调）。

数据流与张量 Shape 速查：
  - observation.image       : (B, 3, 256, 256)   主视角 RGB 图
  - observation.wrist_image : (B, 3, 256, 256)   腕部视角 RGB 图
  - observation.state       : (B, 7)             当前 6 轴关节角 + 夹爪状态
  - action                  : (B, chunk_size, 7) 未来 chunk_size 步目标关节动作块
  - task                    : list[str]          英文自然语言指令

训练主流程：
  DataLoader -> VisualAugmentor -> preprocessor -> policy.forward -> loss.backward
"""

import argparse
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微调 SmolVLA 语言条件跨障分拣策略")
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="lerobot/smolvla_base",
        help="预训练权重 ID 或一阶段本地权重目录（二阶段精调时传入本地目录）",
    )
    parser.add_argument("--repo-id", type=str, default="omy_sorting_vla", help="数据集标识名")
    parser.add_argument("--dataset-root", type=str, default="./dataset/smolvla_dataset", help="本地数据集路径")
    parser.add_argument("--output-dir", type=str, default="./ckpt/smolvla_v2", help="模型与曲线输出目录")

    parser.add_argument("--chunk-size", type=int, default=30, help="每次预测的动作块长度（20Hz 下 30 步 = 1.5s）")
    parser.add_argument("--n-action-steps", type=int, default=15, help="推理时每次连续执行的动作步数")
    parser.add_argument("--steps", type=int, default=10000, help="总训练迭代步数")
    parser.add_argument("--batch-size", type=int, default=16, help="单步训练批次大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率（默认沿用模型配置 1e-4）")
    parser.add_argument("--warmup-steps", type=int, default=500, help="线性 Warmup 步数")

    parser.add_argument(
        "--freeze-vision-encoder",
        dest="freeze_vision_encoder",
        action="store_true",
        default=True,
        help="冻结 SigLIP 视觉编码器（默认开启）",
    )
    parser.add_argument(
        "--no-freeze-vision-encoder",
        dest="freeze_vision_encoder",
        action="store_false",
        help="解冻视觉编码器一起微调",
    )
    parser.add_argument(
        "--train-expert-only",
        dest="train_expert_only",
        action="store_true",
        default=True,
        help="一阶段：冻结 VLM 主干仅微调动作专家（默认开启）",
    )
    parser.add_argument(
        "--no-train-expert-only",
        dest="train_expert_only",
        action="store_false",
        help="二阶段：解冻 VLM 主干与动作专家联合精调",
    )

    parser.add_argument(
        "--image-aug",
        action="store_true",
        help="对主视角开启视觉增强（亮度抖动 0.2 + 高斯噪声 0.02 + 随机遮挡 0.15）",
    )
    parser.add_argument("--aug-wrist", action="store_true", help="同时对腕部视角施加 0.5x 增强")
    parser.add_argument("--val-episodes", type=int, default=4, help="留出最后 N 条轨迹作为验证集")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log-freq", type=int, default=50, help="日志打印间隔步数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取数据集元信息并拆分输入/输出特征
    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    # 2. 加载 SmolVLA 配置、权重及本地统计量预/后处理器
    cfg = SmolVLAConfig.from_pretrained(args.pretrained_path)
    cfg.input_features = input_features
    cfg.output_features = output_features
    cfg.chunk_size = args.chunk_size
    cfg.n_action_steps = args.n_action_steps
    cfg.device = device.type
    cfg.freeze_vision_encoder = args.freeze_vision_encoder
    cfg.train_expert_only = args.train_expert_only

    policy = SmolVLAPolicy.from_pretrained(args.pretrained_path, config=cfg)
    policy.to(device).train()

    # 注意：不传 pretrained_path，确保使用当前本地数据集统计量做归一化
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, dataset_stats=dataset_metadata.stats
    )

    # 3. 按整条 Episode 切分训练集与留出验证集
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    full_dataset = LeRobotDataset(
        args.repo_id, delta_timestamps=delta_timestamps, root=args.dataset_root
    )
    train_dataset, val_ep_ids = split_train_val_episodes(full_dataset, args.val_episodes)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    augmentor = VisualAugmentor(
        color_jitter=0.2 if args.image_aug else 0.0,
        noise_std=0.02 if args.image_aug else 0.0,
        erasing_prob=0.15 if args.image_aug else 0.0,
        augment_wrist=args.aug_wrist,
    )

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else cfg.optimizer_lr
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer, total_steps=args.steps, warmup_steps=args.warmup_steps
    )
    logger = MetricsLogger(save_dir)

    n_train, n_total = count_parameters(policy)
    print(
        f"[Train SmolVLA] pretrained={args.pretrained_path} | device={device.type} | "
        f"trainable={n_train / 1e6:.1f}M/{n_total / 1e6:.1f}M | "
        f"expert_only={args.train_expert_only} | chunk={args.chunk_size}/{args.n_action_steps} | "
        f"val_eps={val_ep_ids} | image_aug={args.image_aug} (wrist={args.aug_wrist})"
    )

    # 4. 标准 PyTorch 训练主循环
    data_iter = iter(dataloader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = augmentor(batch)
        inp_batch = preprocessor(batch)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy.forward(inp_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_freq == 0 or step == args.steps:
            cur_lr = float(scheduler.get_last_lr()[0])
            loss_val = float(loss.detach().item())
            logger.log(step=step, loss=loss_val, lr=cur_lr)
            print(f"step {step:6d}/{args.steps} | loss: {loss_val:.4f} | lr: {cur_lr:.2e}")

    # 5. 保存权重、训练曲线并在留出验证集上做开环 MAE 评估
    save_artifacts(save_dir, policy, preprocessor, postprocessor)
    logger.save_plot(save_dir / "loss_curve.png")
    print(f"[Save] Saved SmolVLA checkpoint & loss curve to: {save_dir}")

    eval_eps = val_ep_ids if val_ep_ids else [0]
    evaluate_heldout_mae(
        policy=policy,
        dataset=full_dataset,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        episode_indices=eval_eps,
        save_plot_path=save_dir / f"eval_val_ep_{eval_eps[0]}.png",
    )


if __name__ == "__main__":
    main()
