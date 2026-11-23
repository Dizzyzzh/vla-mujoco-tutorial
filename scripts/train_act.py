import argparse  # 用于解析命令行开关参数
from pathlib import Path  # 用于跨平台路径拼接与目录创建
from typing import Optional  # 可选类型标注

import matplotlib.pyplot as plt  # 用于绘制预测动作与真值动作的对比曲线
import torch  # PyTorch 深度学习框架

from lerobot.configs.types import FeatureType  # 特征类型枚举（如 ACTION、STATE、VISUAL 等）
from lerobot.datasets.factory import resolve_delta_timestamps  # 根据策略配置解析动作窗口的时间偏移量
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # 数据集本体与元信息加载类
from lerobot.policies.act.configuration_act import ACTConfig  # ACT 算法的配置类
from lerobot.policies.act.modeling_act import ACTPolicy  # ACT 策略网络模型类
from lerobot.policies.factory import make_pre_post_processors  # LeRobot 0.7.0 预处理/后处理器工厂函数
from lerobot.utils.feature_utils import dataset_to_policy_features  # 将数据集特征格式转换为策略所需特征字典

# ==================== 默认训练配置（可直接修改） ====================
repo_id = "omy_pnp"  # 数据集名称
dataset_root = "./dataset/ACT_dataset"  # 本地数据集根目录
output_dir = Path("./ckpt/act_y")  # 模型与处理器权重保存目录

chunk_size = 10  # 每次前向预测未来 10 个时间步的动作块（Action Chunk）
n_action_steps = 10  # 推理时每次连续执行预测块中的 10 步动作
batch_size = 64  # 每个训练批次的样本数
training_steps = 50000  # 总训练迭代步数
target_loss: Optional[float] = 0.03  # 目标 loss 阈值（loss <= 该值时提前停止训练并保存；None 表示不启用）
log_freq = 10  # 每隔 10 步打印一次训练日志
image_noise_std = 0.02  # 主视角图像高斯噪声标准差（0 表示关闭）

# 模型评估开关配置
enable_eval = True  # True=训练完成后执行离线开环推理评估；False=关闭评估
eval_episode_index = 0  # 评估时使用的轨迹编号（从 0 开始）
eval_show_plot = True  # 是否弹出预测动作与真值动作对比曲线窗口（无图形界面环境可设为 False）
eval_save_plot = True  # 是否将评估对比图保存至 output_dir 目录


class EpisodeSampler(torch.utils.data.Sampler):
    """按时间顺序采样指定 episode 的所有帧索引，用于离线开环评估。"""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        ep = dataset.meta.episodes[episode_index]  # 获取指定 episode 的元信息字典
        from_idx = int(ep["dataset_from_index"])  # 该 episode 在整个数据集中的起始帧索引
        to_idx = int(ep["dataset_to_index"])  # 该 episode 在整个数据集中的结束帧索引（不含）
        self.frame_ids = range(from_idx, to_idx)  # 生成按时间顺序排列的连续帧索引范围

    def __iter__(self):
        return iter(self.frame_ids)  # 返回帧索引迭代器，供 DataLoader 顺序读取

    def __len__(self) -> int:
        return len(self.frame_ids)  # 返回该 episode 的总帧数


def evaluate_policy(
    policy: ACTPolicy,
    dataset: LeRobotDataset,
    preprocessor,
    postprocessor,
    device: torch.device,
    episode_index: int = 0,
    show_plot: bool = True,
    save_plot_path: Optional[Path] = None,
) -> float:
    """对指定 episode 执行离线开环推理评估，并按需可视化或保存对比图。"""
    policy.eval()  # 将模型切换为评估模式（关闭 Dropout 等随机层）
    policy.reset()  # 重置策略内部维护的动作块缓存队列
    actions = []  # 收集模型预测出的每一步动作
    gt_actions = []  # 收集数据集中的真实动作（Ground Truth）

    episode_sampler = EpisodeSampler(dataset, episode_index=episode_index)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # 主进程单线程读取
        batch_size=1,  # 逐帧推理（模拟在线控制每步输入 1 帧）
        shuffle=False,  # 必须关闭打乱，保持时间顺序
        pin_memory=device.type != "cpu",
        sampler=episode_sampler,
    )

    with torch.no_grad():  # 推理阶段关闭梯度计算以节省显存并加速
        for batch in test_dataloader:
            gt_actions.append(batch["action"][:, 0, :].cpu())  # 取当前帧真实动作
            inp_batch = preprocessor(batch)  # 对当前帧观测进行归一化并迁移至计算设备
            pred_action = policy.select_action(inp_batch)  # 输出当前步的归一化预测动作
            pred_action = postprocessor(pred_action)  # 反归一化回原始物理尺度
            actions.append(pred_action.cpu())

    actions = torch.cat(actions, dim=0)
    gt_actions = torch.cat(gt_actions, dim=0)
    mean_action_error = torch.mean(torch.abs(actions - gt_actions)).item()  # 计算整条轨迹的平均绝对误差（MAE）
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
    parser = argparse.ArgumentParser(description="训练 ACT 策略并按需执行离线开环模型评估")
    parser.add_argument("--dataset-root", type=str, default=dataset_root, help="本地数据集根目录")
    parser.add_argument("--output-dir", type=str, default=str(output_dir), help="模型与处理器权重保存目录")
    parser.add_argument("--steps", type=int, default=training_steps, help="总训练迭代步数")
    parser.add_argument("--batch-size", type=int, default=batch_size, help="训练批次大小")
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
    save_dir.mkdir(parents=True, exist_ok=True)  # 递归创建输出目录，若已存在则忽略

    # 1. 读取数据集元信息并构建输入/输出特征
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 2. 实例化 ACT 配置、模型与 pre/post processor
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        device=device.type,
    )
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)  # 解析动作窗口所需相对时间戳

    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=dataset_metadata.stats,  # 传入数据集统计量（均值、标准差等）用于归一化/反归一化
    )

    # 3. 加载数据集、优化器与 DataLoader
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, root=args.dataset_root)
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=cfg.optimizer_lr)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows 下建议保持 0，避免多进程 PicklingError
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # 4. 训练循环（可自由在此插入后训练逻辑：如样本过滤、自定义 Loss 等）
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            if image_noise_std > 0 and "observation.image" in batch:
                # 在归一化前给主视角图像加轻量高斯噪声增强鲁棒性
                img = batch["observation.image"]
                batch["observation.image"] = (img + torch.randn_like(img) * image_noise_std).clamp(0, 1)

            inp_batch = preprocessor(batch)  # 执行归一化并自动迁移到目标设备
            loss, _ = policy.forward(inp_batch)  # 前向计算 ACT 损失（动作重构损失 + CVAE KL 散度）
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss_val:.3f}")
            if args.target_loss is not None and loss_val <= args.target_loss:
                print(f"step: {step} loss: {loss_val:.4f} <= {args.target_loss}，达到目标 loss，停止训练并保存模型")
                done = True
                break
            step += 1
            if step >= args.steps:
                done = True
                break

    # 5. 保存模型权重与 pre/post processor（部署时反归一化必需）
    policy.save_pretrained(save_dir)
    preprocessor.save_pretrained(save_dir)
    postprocessor.save_pretrained(save_dir)
    print(f"[Save] Saved ACT policy & processors to: {save_dir}")

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
