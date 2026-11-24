import json
import math
import random
from pathlib import Path
from typing import Optional

import matplotlib

# 使用无界面后端，防止在云服务器或后台脚本中绘图弹窗阻塞进程
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def set_seed(seed: int) -> None:
    """统一固定 Python、NumPy 与 PyTorch 随机种子以保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """统计模型的可训练参数量与总参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def save_artifacts(save_dir: Path, policy, preprocessor, postprocessor) -> None:
    """保存模型权重配置以及 pre/post processor（闭环评测与部署反归一化必需）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.025,
) -> torch.optim.lr_scheduler.LambdaLR:
    """构造线性 Warmup + 余弦退火学习率调度器。"""
    start_ratio = max(1e-3, min_lr_ratio)

    def lr_lambda(current_step: int) -> float:
        if total_steps <= 1:
            return 1.0
        eff_warmup = min(max(warmup_steps, 0), total_steps - 1) if warmup_steps > 0 else 0
        if eff_warmup > 0 and current_step < eff_warmup:
            progress = current_step / float(eff_warmup)
            return start_ratio + (1.0 - start_ratio) * progress
        decay_steps = max(1, total_steps - eff_warmup)
        progress = min(1.0, max(0.0, (current_step - eff_warmup) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def split_train_val_episodes(
    dataset: LeRobotDataset, val_episodes: int
) -> tuple[torch.utils.data.Dataset, list[int]]:
    """
    按 episode 维度留出最后 val_episodes 条轨迹作为无泄露验证集，
    返回 (train_subset, val_episode_ids)。
    """
    total_eps = int(dataset.meta.total_episodes)
    if val_episodes <= 0 or val_episodes >= total_eps:
        return dataset, []

    train_end_ep = total_eps - val_episodes
    val_ep_ids = list(range(train_end_ep, total_eps))
    train_indices: list[int] = []
    for ep_idx in range(train_end_ep):
        ep = dataset.meta.episodes[ep_idx]
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])
        train_indices.extend(range(from_idx, to_idx))

    return torch.utils.data.Subset(dataset, train_indices), val_ep_ids


class VisualAugmentor:
    """
    面向模仿学习的轻量化向量化图像增强器（仅在训练阶段、归一化前施加于 [0, 1] 张量）。
    用于提升策略对环境光照变化、背景纹理及 OOD 干扰物的鲁棒性。
    """

    def __init__(
        self,
        color_jitter: float = 0.0,
        noise_std: float = 0.0,
        erasing_prob: float = 0.0,
        augment_wrist: bool = False,
    ) -> None:
        self.color_jitter = float(color_jitter)
        self.noise_std = float(noise_std)
        self.erasing_prob = float(erasing_prob)
        self.augment_wrist = bool(augment_wrist)

    @property
    def enabled(self) -> bool:
        return self.color_jitter > 0 or self.noise_std > 0 or self.erasing_prob > 0

    def _augment_tensor(self, img: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        squeezed = False
        if img.ndim == 3:
            img = img.unsqueeze(0)
            squeezed = True

        out = img.clone()
        bsz, _, height, width = out.shape
        device, dtype = out.device, out.dtype

        jitter = self.color_jitter * scale
        if jitter > 0:
            # 1) 亮度抖动
            b_scale = 1.0 + (
                torch.rand((bsz, 1, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0
            ) * jitter
            out = out * b_scale

            # 2) 对比度抖动
            c_scale = 1.0 + (
                torch.rand((bsz, 1, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0
            ) * jitter
            mean = out.mean(dim=(-2, -1), keepdim=True)
            out = (out - mean) * c_scale + mean

            # 3) 饱和度抖动
            s_scale = 1.0 + (
                torch.rand((bsz, 1, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0
            ) * jitter
            luma = (
                0.2989 * out[:, 0:1]
                + 0.5870 * out[:, 1:2]
                + 0.1140 * out[:, 2:3]
            )
            out = luma + (out - luma) * s_scale

        noise_std = self.noise_std * scale
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std

        erasing_p = self.erasing_prob * scale
        if erasing_p > 0:
            area = height * width
            for b in range(bsz):
                if float(torch.rand(1).item()) < erasing_p:
                    target_area = float(torch.empty(1).uniform_(0.02, 0.15).item()) * area
                    aspect = float(torch.empty(1).uniform_(0.3, 3.3).item())
                    eh = min(height, max(1, int(round(math.sqrt(target_area * aspect)))))
                    ew = min(width, max(1, int(round(math.sqrt(target_area / aspect)))))
                    top = int(torch.randint(0, max(1, height - eh + 1), (1,)).item())
                    left = int(torch.randint(0, max(1, width - ew + 1), (1,)).item())
                    fill = torch.rand((3, 1, 1), device=device, dtype=dtype)
                    out[b, :, top : top + eh, left : left + ew] = fill

        out.clamp_(0.0, 1.0)
        return out.squeeze(0) if squeezed else out

    def __call__(self, batch: dict) -> dict:
        if not self.enabled:
            return batch
        if "observation.image" in batch:
            batch["observation.image"] = self._augment_tensor(
                batch["observation.image"], scale=1.0
            )
        if self.augment_wrist and "observation.wrist_image" in batch:
            batch["observation.wrist_image"] = self._augment_tensor(
                batch["observation.wrist_image"], scale=0.5
            )
        return batch


class MetricsLogger:
    """记录训练指标到 JSONL 并生成 Loss 与学习率曲线图。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "train_metrics.jsonl"
        self.history: list[dict] = []
        # 每次新建日志时清空旧文件，防止不同轮次指标混淆
        self.jsonl_path.write_text("", encoding="utf-8")

    def log(
        self,
        step: int,
        loss: float,
        ema_loss: float,
        lr: float,
        grad_norm: float,
        step_ms: float,
        vram_mb: float,
    ) -> None:
        record = {
            "step": int(step),
            "loss": round(float(loss), 6),
            "ema_loss": round(float(ema_loss), 6),
            "lr": float(lr),
            "grad_norm": round(float(grad_norm), 4),
            "step_ms": round(float(step_ms), 2),
            "vram_mb": round(float(vram_mb), 1),
        }
        self.history.append(record)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_plot(self, save_path: Path) -> None:
        if not self.history:
            return
        steps = [r["step"] for r in self.history]
        losses = [r["loss"] for r in self.history]
        ema_losses = [r["ema_loss"] for r in self.history]
        lrs = [r["lr"] for r in self.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(steps, losses, alpha=0.35, color="#4C78A8", label="Raw Loss")
        ax1.plot(steps, ema_losses, linewidth=2.0, color="#E45756", label="EMA Loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss & Learning Rate Schedule")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(steps, lrs, linewidth=1.8, color="#54A24B", label="Learning Rate")
        ax2.set_xlabel("Optimizer Step")
        ax2.set_ylabel("Learning Rate")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


class EpisodeSampler(torch.utils.data.Sampler):
    """按时间顺序采样指定 episode 的所有帧索引，用于离线开环评估。"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int) -> None:
        ep = dataset.meta.episodes[episode_index]
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def evaluate_heldout_mae(
    policy,
    dataset: LeRobotDataset,
    preprocessor,
    postprocessor,
    device: torch.device,
    episode_indices: list[int],
    save_plot_path: Optional[Path] = None,
    show_plot: bool = False,
) -> float:
    """在指定（或留出验证集）episode 上执行离线开环评估，返回平均 MAE 并保存首条对比图。"""
    if not episode_indices:
        episode_indices = [0]

    policy.eval()
    ep_maes: list[float] = []

    for idx, ep_idx in enumerate(episode_indices):
        policy.reset()
        actions = []
        gt_actions = []

        sampler = EpisodeSampler(dataset, episode_index=ep_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=0,
            batch_size=1,
            shuffle=False,
            pin_memory=device.type != "cpu",
            sampler=sampler,
        )

        with torch.inference_mode():
            for batch in dataloader:
                gt_action = (
                    batch["action"][:, 0, :] if batch["action"].ndim == 3 else batch["action"]
                )
                gt_actions.append(gt_action.cpu())
                inp_batch = preprocessor(batch)
                pred_action = policy.select_action(inp_batch)
                pred_action = postprocessor(pred_action)
                actions.append(pred_action.cpu())

        pred_cat = torch.cat(actions, dim=0)
        gt_cat = torch.cat(gt_actions, dim=0)
        mae = torch.mean(torch.abs(pred_cat - gt_cat)).item()
        ep_maes.append(mae)
        print(f"[Eval] Episode {ep_idx} Mean action error: {mae:.4f}")

        if idx == 0 and (save_plot_path is not None or show_plot):
            action_dim = pred_cat.shape[-1]
            fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10), sharex=True)
            if action_dim == 1:
                axs = [axs]
            for d in range(action_dim):
                axs[d].plot(pred_cat[:, d].numpy(), label="pred")
                axs[d].plot(gt_cat[:, d].numpy(), label="gt")
                axs[d].set_ylabel(f"dim_{d}")
                axs[d].legend(loc="upper right")
            axs[0].set_title(f"Open-Loop Action Prediction vs GT (Episode {ep_idx}, MAE={mae:.4f})")
            plt.tight_layout()
            if save_plot_path is not None:
                plt.savefig(save_plot_path, dpi=150)
                print(f"[Eval] Saved evaluation plot to: {save_plot_path}")
            if show_plot:
                plt.show()
            else:
                plt.close(fig)

    avg_mae = float(np.mean(ep_maes))
    print(f"[Eval] Average MAE across {len(ep_maes)} episode(s): {avg_mae:.4f}")
    return avg_mae
