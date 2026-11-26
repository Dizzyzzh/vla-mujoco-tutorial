"""
训练辅助工具模块：
- 随机种子固定、参数量统计、权重与预/后处理器保存
- 按整条 Episode 切分训练/验证集（防止相邻帧数据泄露）
- 兼容 4D/5D 张量的轻量视觉增强（亮度抖动 + 高斯噪声 + 随机遮挡）
- 训练 Loss / 学习率记录绘图与留出验证集开环 MAE 评估
"""

import json
import math
import random
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子以保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """返回模型的 (可训练参数量, 总参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def save_artifacts(save_dir: Path, policy, preprocessor, postprocessor) -> None:
    """保存模型权重及配套 pre/post processor（推理反归一化必需）。"""
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
    """线性 Warmup + 余弦退火学习率调度器。"""
    start_ratio = max(1e-3, min_lr_ratio)

    def lr_lambda(current_step: int) -> float:
        if total_steps <= 1:
            return 1.0
        eff_warmup = min(max(warmup_steps, 0), total_steps - 1) if warmup_steps > 0 else 0
        if eff_warmup > 0 and current_step < eff_warmup:
            return start_ratio + (1.0 - start_ratio) * (current_step / float(eff_warmup))
        decay_steps = max(1, total_steps - eff_warmup)
        progress = min(1.0, max(0.0, (current_step - eff_warmup) / float(decay_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def split_train_val_episodes(
    dataset: LeRobotDataset, val_episodes: int
) -> tuple[torch.utils.data.Dataset, list[int]]:
    """按整条 Episode 留出最后 val_episodes 条作为验证集，避免相邻帧泄露。"""
    total_eps = int(dataset.meta.total_episodes)
    if val_episodes <= 0 or val_episodes >= total_eps:
        return dataset, []

    train_end_ep = total_eps - val_episodes
    val_ep_ids = list(range(train_end_ep, total_eps))
    train_indices: list[int] = []
    for ep_idx in range(train_end_ep):
        ep = dataset.meta.episodes[ep_idx]
        train_indices.extend(range(int(ep["dataset_from_index"]), int(ep["dataset_to_index"])))

    return torch.utils.data.Subset(dataset, train_indices), val_ep_ids


class VisualAugmentor:
    """
    轻量图像增强（在归一化前作用于 [0, 1] 图像张量）：
    通过展平前缀维度同时兼容 4D (B, C, H, W) 与 5D (B, T, C, H, W) 输入。
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
        if img.ndim < 3:
            raise ValueError(f"Expected image tensor with ndim >= 3, got shape {tuple(img.shape)}")
        orig_shape = img.shape
        channels, height, width = orig_shape[-3:]
        # 展平前缀维度（如 B 或 B*T），统一按 (N, C, H, W) 增强后再还原
        out = img.reshape(-1, channels, height, width).clone()
        bsz, device, dtype = out.shape[0], out.device, out.dtype

        jitter = self.color_jitter * scale
        if jitter > 0:
            b_scale = 1.0 + (torch.rand((bsz, 1, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0) * jitter
            out = out * b_scale

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
                    out[b, :, top : top + eh, left : left + ew] = torch.rand(
                        (channels, 1, 1), device=device, dtype=dtype
                    )

        return out.clamp_(0.0, 1.0).reshape(orig_shape)

    def __call__(self, batch: dict) -> dict:
        if not self.enabled:
            return batch
        if "observation.image" in batch:
            batch["observation.image"] = self._augment_tensor(batch["observation.image"], scale=1.0)
        if self.augment_wrist and "observation.wrist_image" in batch:
            batch["observation.wrist_image"] = self._augment_tensor(
                batch["observation.wrist_image"], scale=0.5
            )
        return batch


class MetricsLogger:
    """记录训练步数、Loss 与学习率到 JSONL，并在训练结束后保存曲线图。"""

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = output_dir / "train_metrics.jsonl"
        self.history: list[dict] = []
        self.jsonl_path.write_text("", encoding="utf-8")

    def log(self, step: int, loss: float, lr: float) -> None:
        record = {"step": int(step), "loss": round(float(loss), 6), "lr": float(lr)}
        self.history.append(record)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_plot(self, save_path: Path) -> None:
        if not self.history:
            return
        steps = [r["step"] for r in self.history]
        losses = [r["loss"] for r in self.history]
        lrs = [r["lr"] for r in self.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(steps, losses, linewidth=1.8, color="#E45756", label="Train Loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss & Learning Rate")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(steps, lrs, linewidth=1.8, color="#54A24B", label="Learning Rate")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


class EpisodeSampler(torch.utils.data.Sampler):
    """按时间顺序采样单条 Episode 的全部帧，用于离线开环评估。"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int) -> None:
        ep = dataset.meta.episodes[episode_index]
        self.frame_ids = range(int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))

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
) -> float:
    """在留出验证集 Episode 上运行离线开环预测，返回平均 MAE 并保存对比图。"""
    episode_indices = episode_indices or [0]
    policy.eval()
    ep_maes: list[float] = []

    for idx, ep_idx in enumerate(episode_indices):
        policy.reset()
        preds, gts = [], []
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type != "cpu",
            sampler=EpisodeSampler(dataset, episode_index=ep_idx),
        )

        with torch.inference_mode():
            for batch in loader:
                gt = batch["action"][:, 0, :] if batch["action"].ndim == 3 else batch["action"]
                gts.append(gt.float().cpu())
                pred = postprocessor(policy.select_action(preprocessor(batch)))
                preds.append(pred.float().cpu())

        pred_cat = torch.cat(preds, dim=0)
        gt_cat = torch.cat(gts, dim=0)
        mae = torch.mean(torch.abs(pred_cat - gt_cat)).item()
        ep_maes.append(mae)
        print(f"[Eval] Episode {ep_idx} MAE: {mae:.4f}")

        if idx == 0 and save_plot_path is not None:
            action_dim = pred_cat.shape[-1]
            fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10), sharex=True)
            axs = [axs] if action_dim == 1 else axs
            for d in range(action_dim):
                axs[d].plot(pred_cat[:, d].numpy(), label="pred")
                axs[d].plot(gt_cat[:, d].numpy(), label="gt")
                axs[d].set_ylabel(f"dim_{d}")
                axs[d].legend(loc="upper right")
            axs[0].set_title(f"Open-Loop Prediction vs GT (Episode {ep_idx}, MAE={mae:.4f})")
            plt.tight_layout()
            plt.savefig(save_plot_path, dpi=150)
            plt.close(fig)

    avg_mae = float(np.mean(ep_maes))
    print(f"[Eval] Held-out Average MAE ({len(ep_maes)} eps): {avg_mae:.4f}")
    return avg_mae
