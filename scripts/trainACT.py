from pathlib import Path  # 用于跨平台路径拼接与目录创建

import matplotlib.pyplot as plt  # 用于绘制预测动作与真值动作的对比曲线
import torch  # PyTorch 深度学习框架

from lerobot.configs.types import FeatureType  # 特征类型枚举（如 ACTION、STATE、VISUAL 等）
from lerobot.datasets.factory import resolve_delta_timestamps  # 根据策略配置解析动作窗口的时间偏移量
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # 数据集本体与元信息加载类
from lerobot.policies.act.configuration_act import ACTConfig  # ACT 算法的配置类
from lerobot.policies.act.modeling_act import ACTPolicy  # ACT 策略网络模型类
from lerobot.policies.factory import make_pre_post_processors  # LeRobot 0.7.0 预处理/后处理器工厂函数
from lerobot.utils.feature_utils import dataset_to_policy_features  # 将数据集特征格式转换为策略所需特征字典

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_steps = 1000  # 总训练迭代步数
log_freq = 10  # 每隔 10 步打印一次训练日志
dataset_root = "./demo_data_example"  # 本地数据集根目录
output_dir = Path("./ckpt/act_y")  # 模型与处理器权重保存目录
output_dir.mkdir(parents=True, exist_ok=True)  # 递归创建输出目录，若已存在则忽略

# 1. 读取数据集元信息并构建输入/输出特征
dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root=dataset_root)  # 仅加载元信息（统计量、特征定义、episode 索引等）
features = dataset_to_policy_features(dataset_metadata.features)  # 转换数据集特征为策略标准特征格式
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}  # 提取动作特征作为策略输出
input_features = {key: ft for key, ft in features.items() if key not in output_features}  # 其余观测/状态特征作为策略输入
input_features.pop("observation.wrist_image", None)  # 剔除腕部相机图像，仅保留主视角图像和状态输入

cfg = ACTConfig(
    input_features=input_features,  # 策略输入特征定义
    output_features=output_features,  # 策略输出特征定义
    chunk_size=10,  # 每次前向预测未来 10 个时间步的动作块（Action Chunk）
    n_action_steps=10,  # 推理时每次连续执行预测块中的 10 步动作
    device=device.type,  # 指定配置中的设备类型（"cuda" 或 "cpu"）
)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)  # 计算读取未来 chunk_size 帧动作所需的相对时间戳

policy = ACTPolicy(cfg)  # 根据配置实例化 ACT 策略模型
policy.train()  # 将模型切换为训练模式
policy.to(device)  # 将模型参数和缓冲区迁移到目标设备

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,  # 传入策略配置
    dataset_stats=dataset_metadata.stats,  # 传入数据集统计量（均值、标准差等），用于归一化/反归一化
)

# 3. 加载数据集与 DataLoader
dataset = LeRobotDataset("omy_pnp", delta_timestamps=delta_timestamps, root=dataset_root)  # 加载完整数据集，并按时间偏移返回动作序列
optimizer = torch.optim.Adam(policy.get_optim_params(), lr=cfg.optimizer_lr)  # 使用 Adam 优化器，学习率取自 ACTConfig 默认配置
dataloader = torch.utils.data.DataLoader(
    dataset,  # 训练数据集
    batch_size=64,  # 每个训练批次包含 64 个样本
    shuffle=True,  # 每个 epoch 随机打乱样本顺序
    pin_memory=device.type != "cpu",  # 使用 GPU 时开启锁页内存以加速主机到设备的数据传输
    drop_last=True,  # 丢弃最后一个不足 batch_size 的不完整批次
)

# 4. 训练循环
step = 0  # 初始化当前训练步数计数器
done = False  # 训练结束标志位
while not done:  # 外层循环确保在数据集遍历完后能继续迭代直到达到 training_steps
    for batch in dataloader:  # 从 DataLoader 中逐批次取出数据
        # 在归一化前给主视角图像加轻量高斯噪声
        img = batch["observation.image"]  # 取出主视角图像张量（像素值通常在 [0, 1]）
        batch["observation.image"] = (img + torch.randn_like(img) * 0.02).clamp(0, 1)  # 叠加标准差 0.02 的高斯噪声并裁剪回 [0, 1]

        inp_batch = preprocessor(batch)  # 对输入执行归一化，并将张量自动迁移到目标设备
        loss, _ = policy.forward(inp_batch)  # 前向计算 ACT 损失（含动作重构损失与 CVAE KL 散度），忽略附加信息 dict
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 清空梯度，准备下一步迭代

        if step % log_freq == 0:  # 每隔 log_freq 步打印一次当前损失
            print(f"step: {step} loss: {loss.item():.3f}")  # 输出当前步数与标量 loss（保留三位小数）
        step += 1  # 步数加 1
        if step >= training_steps:  # 判断是否达到设定的总训练步数
            done = True  # 标记训练完成
            break  # 跳出当前 epoch 的 DataLoader 循环

# 5. 保存模型权重与 pre/post processor（部署时反归一化必需）
policy.save_pretrained(output_dir)  # 保存模型结构配置（config.json）与权重（model.safetensors）
preprocessor.save_pretrained(output_dir)  # 保存输入预处理器配置与统计量
postprocessor.save_pretrained(output_dir)  # 保存输出后处理器配置与统计量，推理还原真实动作尺度时必须加载



# class EpisodeSampler(torch.utils.data.Sampler):

#     def __init__(self, dataset: LeRobotDataset, episode_index: int):
#         ep = dataset.meta.episodes[episode_index]  # 获取指定 episode 的元信息字典
#         from_idx = int(ep["dataset_from_index"])  # 该 episode 在整个数据集中的起始帧索引
#         to_idx = int(ep["dataset_to_index"])  # 该 episode 在整个数据集中的结束帧索引（不含）
#         self.frame_ids = range(from_idx, to_idx)  # 生成按时间顺序排列的连续帧索引范围

#     def __iter__(self):
#         return iter(self.frame_ids)  # 返回帧索引迭代器，供 DataLoader 顺序读取

#     def __len__(self) -> int:
#         return len(self.frame_ids)  # 返回该 episode 的总帧数

# # 6. 离线开环推理验证
# policy.eval()  # 将模型切换为评估模式（关闭 Dropout 等随机层）
# policy.reset()  # 重置策略内部维护的动作块缓存队列
# actions = []  # 用于收集模型预测出的每一步动作
# gt_actions = []  # 用于收集数据集中的真实动作（Ground Truth）
# episode_sampler = EpisodeSampler(dataset, episode_index=0)  # 实例化采样器，按顺序读取第 0 条轨迹
# test_dataloader = torch.utils.data.DataLoader(
#     dataset,  # 复用同一数据集
#     num_workers=0,  # 主进程单线程读取
#     batch_size=1,  # 逐帧推理（模拟在线控制每步输入 1 帧）
#     shuffle=False,  # 必须关闭打乱，保持时间顺序
#     pin_memory=device.type != "cpu",  # GPU 推理时开启锁页内存
#     sampler=episode_sampler,  # 使用自定义顺序采样器
# )

# with torch.no_grad():  # 推理阶段关闭梯度计算以节省显存并加速
#     for batch in test_dataloader:  # 按时间步逐帧遍历第 0 条 episode
#         gt_actions.append(batch["action"][:, 0, :].cpu())  # batch["action"] 形状为 (1, chunk_size, action_dim)，取第 0 步即当前帧真实动作
#         inp_batch = preprocessor(batch)  # 对当前帧观测进行归一化并迁移至计算设备
#         pred_action = policy.select_action(inp_batch)  # 策略输出当前步的归一化预测动作（内部每 n_action_steps 步预测一次 chunk）
#         pred_action = postprocessor(pred_action)  # 将归一化动作反归一化回原始物理尺度
#         actions.append(pred_action.cpu())  # 移回 CPU 并保存预测结果

# actions = torch.cat(actions, dim=0)  # 将所有时间步的预测动作拼接为 (T, action_dim) 张量
# gt_actions = torch.cat(gt_actions, dim=0)  # 将所有时间步的真实动作拼接为 (T, action_dim) 张量
# print(f"Mean action error: {torch.mean(torch.abs(actions - gt_actions)).item():.3f}")  # 计算并打印整条轨迹的平均绝对误差（MAE）

# # 7. 可视化预测动作与真值动作对比
# action_dim = actions.shape[-1]  # 获取动作维度（例如机械臂各关节自由度数量）
# fig, axs = plt.subplots(action_dim, 1, figsize=(10, 10))  # 创建 action_dim 行、1 列的子图布局
# for i in range(action_dim):  # 遍历动作的每一个维度
#     axs[i].plot(actions[:, i].numpy(), label="pred")  # 绘制第 i 维的预测动作曲线
#     axs[i].plot(gt_actions[:, i].numpy(), label="gt")  # 绘制第 i 维的真实动作曲线
#     axs[i].legend()  # 显示图例
# plt.tight_layout()  # 自动调整子图间距，避免标签重叠
# plt.show()  # 弹出窗口显示对比图


