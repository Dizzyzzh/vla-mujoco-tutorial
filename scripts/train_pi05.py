import argparse  # 用于解析命令行开关参数
from contextlib import nullcontext  # 用于按需启用混合精度上下文
from pathlib import Path  # 用于跨平台路径拼接与目录创建
from typing import Optional  # 可选类型标注

import matplotlib.pyplot as plt  # 用于绘制预测动作与真值动作的对比曲线
import torch  # PyTorch 深度学习框架

from lerobot.configs.types import FeatureType  # 特征类型枚举（如 ACTION、STATE、VISUAL 等）
from lerobot.datasets.factory import resolve_delta_timestamps  # 根据策略配置解析动作窗口的时间偏移量
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # 数据集本体与元信息加载类
from lerobot.policies.factory import make_pre_post_processors  # LeRobot 0.7.0 预处理/后处理器工厂函数
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # PI0.5 配置类
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # PI0.5 策略模型类
from lerobot.utils.feature_utils import dataset_to_policy_features  # 将数据集特征格式转换为策略所需特征字典

# ==================== 默认训练配置（可直接修改） ====================
repo_id = "omy_pnp_language"  # 数据集 repo_id
dataset_root = "./demo_data_language"  # 本地数据集根目录
output_dir = Path("./ckpt/pi05_omy")  # 模型与处理器权重保存目录
pretrained_path: Optional[str] = "lerobot/pi05_base"  # 预训练权重或本地 checkpoint（后训练可直接改成本地目录）

chunk_size = 5  # 每次前向预测的动作块长度
n_action_steps = 5  # 推理时每次连续执行的动作步数
batch_size = 8  # 每个训练批次样本数（PI0.5 显存需求较大，按显存调整）
training_steps = 20000  # 总训练迭代步数
log_freq = 50  # 每隔 50 步打印一次训练日志
save_freq = 10000  # 每隔 10000 步保存一次中间 checkpoint（0 表示仅训练结束保存）
grad_clip_norm = 1.0  # 梯度裁剪最大范数（<=0 表示不裁剪）
target_loss: Optional[float] = None  # 目标 loss 阈值（例如改为 0.02 做早停；None 表示关闭）

# PI0.5 微调与显存控制
dtype = torch.bfloat16  # 权重精度（可选 torch.bfloat16 或 torch.float32）
gradient_checkpointing = True  # 开启梯度检查点大幅降低显存占用
freeze_vision_encoder = False  # 是否仅冻结视觉编码器
train_expert_only = False  # 是否冻结 VLM 主干，仅训练动作专家与投影层
use_amp = False  # 是否开启 torch.autocast 自动混合精度

# 模型评估开关配置
enable_eval = True  # True=训练完成后执行离线开环推理评估；False=关闭评估
eval_episode_index = 0  # 评估时使用的轨迹编号（从 0 开始）
eval_show_plot = True  # 是否弹出预测动作与真值动作对比曲线窗口（无图形界面环境可设为 False）
eval_save_plot = True  # 是否将评估对比图保存至 output_dir 目录


def save_artifacts(save_dir: Path, policy: PI05Policy, preprocessor, postprocessor) -> None:
    """保存模型权重配置以及 pre/post processor（部署时反归一化必需）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)


class EpisodeSampler(torch.utils.data.Sampler):
    """按时间顺序采样指定 episode 的所有帧索引，用于离线开环评估。"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        ep = dataset.meta.episodes[episode_index]
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def evaluate_policy(
    policy: PI05Policy,
    dataset: LeRobotDataset,
    preprocessor,
    postprocessor,
    device: torch.device,
    episode_index: int = 0,
    show_plot: bool = True,
    save_plot_path: Optional[Path] = None,
) -> float:
    """对指定 episode 执行离线开环推理评估，并按需可视化或保存对比图。"""
    policy.eval()
    policy.reset()
    actions = []
    gt_actions = []

    episode_sampler = EpisodeSampler(dataset, episode_index=episode_index)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type != "cpu",
        sampler=episode_sampler,
    )

    with torch.no_grad():
        for batch in test_dataloader:
            gt_action = batch["action"][:, 0, :] if batch["action"].ndim == 3 else batch["action"]
            gt_actions.append(gt_action.cpu())
            inp_batch = preprocessor(batch)  # 完成状态/动作分位数归一化、语言分词并迁移至计算设备
            pred_action = policy.select_action(inp_batch)
            pred_action = postprocessor(pred_action)  # 反归一化回原始物理尺度
            actions.append(pred_action.cpu())

    actions = torch.cat(actions, dim=0)
    gt_actions = torch.cat(gt_actions, dim=0)
    mean_action_error = torch.mean(torch.abs(actions - gt_actions)).item()
    print(f"[Eval] Episode {episode_index} Mean action error: {mean_action_error:.3f}")

    if show_plot or save_plot_path is not None:
        action_dim = actions.shape[-1]
        fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10))
        if action_dim == 1:
            axs = [axs]
        for i in range(action_dim):
            axs[i].plot(actions[:, i].numpy(), label="pred")
            axs[i].plot(gt_actions[:, i].numpy(), label="gt")
            axs[i].legend()
        plt.tight_layout()
        if save_plot_path is not None:
            plt.savefig(save_plot_path, dpi=150)
            print(f"[Eval] Saved evaluation plot to: {save_plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return mean_action_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练或后训练 PI0.5 策略并按需执行离线开环评估")
    parser.add_argument("--pretrained-path", type=str, default=pretrained_path, help="预训练权重或本地 checkpoint 路径")
    parser.add_argument("--no-pretrained", action="store_true", help="强制从头初始化训练，不加载预训练权重")
    parser.add_argument("--dataset-root", type=str, default=dataset_root, help="本地数据集根目录")
    parser.add_argument("--repo-id", type=str, default=repo_id, help="数据集 repo_id")
    parser.add_argument("--output-dir", type=str, default=str(output_dir), help="模型与处理器权重保存目录")
    parser.add_argument("--steps", type=int, default=training_steps, help="总训练迭代步数")
    parser.add_argument("--batch-size", type=int, default=batch_size, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 PI05Config 默认学习率")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16"],
        default=str(dtype).removeprefix("torch."),
        help="模型权重精度配置",
    )
    parser.add_argument("--target-loss", type=float, default=target_loss, help="当 loss <= 该值时提前停止训练并保存")
    parser.add_argument("--eval", dest="enable_eval", action="store_true", default=enable_eval, help="开启训练后模型评估")
    parser.add_argument("--no-eval", dest="enable_eval", action="store_false", help="关闭训练后模型评估")
    parser.add_argument("--eval-episode", type=int, default=eval_episode_index, help="评估使用的 episode 编号")
    parser.add_argument("--no-plot", dest="eval_show_plot", action="store_false", default=eval_show_plot, help="关闭评估曲线弹窗")
    parser.add_argument("--save-eval-plot", dest="eval_save_plot", action="store_true", default=eval_save_plot, help="保存评估对比图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    active_pretrained = None if args.no_pretrained else args.pretrained_path
    model_dtype = getattr(torch, args.dtype)

    # 1. 读取数据集元信息并构建输入/输出特征
    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 2. 构建 PI0.5 配置并实例化策略模型
    if active_pretrained:
        # 先加载预训练配置以保留 PaliGemma / Gemma Expert 变体等结构超参，再覆盖当前任务参数
        cfg = PI05Config.from_pretrained(active_pretrained)
        cfg.input_features = input_features
        cfg.output_features = output_features
        cfg.chunk_size = chunk_size
        cfg.n_action_steps = n_action_steps
        cfg.device = device.type
        cfg.dtype = model_dtype
        cfg.gradient_checkpointing = gradient_checkpointing
        cfg.freeze_vision_encoder = freeze_vision_encoder
        cfg.train_expert_only = train_expert_only
        policy = PI05Policy.from_pretrained(active_pretrained, config=cfg)
    else:
        cfg = PI05Config(
            input_features=input_features,
            output_features=output_features,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            device=device.type,
            dtype=model_dtype,
            gradient_checkpointing=gradient_checkpointing,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
        )
        policy = PI05Policy(cfg)

    policy.train()
    policy.to(device)
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

    # PI05Config 默认使用分位数归一化（QUANTILES），此处传入当前数据集统计量生成对应处理器
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=dataset_metadata.stats,
    )

    # 3. 加载数据集、可训练参数优化器与 DataLoader（可在此前自由挂载 LoRA 或自定义冻层）
    dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps, root=args.dataset_root)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else cfg.optimizer_lr
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows 下建议保持 0，避免多进程问题
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
    # bfloat16 无需 GradScaler；仅 float16 混合精度时启用梯度缩放
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled and amp_dtype == torch.float16)
    print(
        f"[Train PI0.5] pretrained={active_pretrained}, dtype={args.dtype}, "
        f"steps={args.steps}, batch_size={args.batch_size}, lr={lr}"
    )

    # 4. 训练循环（后训练可改 reduction="none" 实现逐样本加权或自定义 Loss）
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            inp_batch = preprocessor(batch)
            autocast_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else nullcontext()
            with autocast_ctx:
                loss, _ = policy.forward(inp_batch)  # 如需逐样本 Reward 加权，可改为 reduction="none"

            grad_scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

            loss_val = loss.item()
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss_val:.3f}")

            step += 1
            if save_freq > 0 and step % save_freq == 0 and step < args.steps:
                ckpt_dir = save_dir / f"step_{step}"
                save_artifacts(ckpt_dir, policy, preprocessor, postprocessor)
                print(f"[Save] Saved intermediate checkpoint to: {ckpt_dir}")

            if args.target_loss is not None and loss_val <= args.target_loss:
                print(f"step: {step} loss: {loss_val:.4f} <= {args.target_loss}，达到目标 loss，停止训练并保存模型")
                done = True
                break
            if step >= args.steps:
                done = True
                break

    # 5. 保存最终模型权重与 pre/post processor
    save_artifacts(save_dir, policy, preprocessor, postprocessor)
    print(f"[Save] Saved PI0.5 policy & processors to: {save_dir}")

    # 6. 离线开环推理评估
    if args.enable_eval:
        plot_path = save_dir / f"eval_episode_{args.eval_episode}.png" if args.eval_save_plot else None
        evaluate_policy(
            policy=policy,
            dataset=dataset,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            episode_index=args.eval_episode,
            show_plot=args.eval_show_plot,
            save_plot_path=plot_path,
        )
    else:
        print("[Eval] 模型评估已关闭（可使用 --eval 参数开启）。")


if __name__ == "__main__":
    main()
