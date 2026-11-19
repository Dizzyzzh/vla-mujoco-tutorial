import numpy as np  # 数值计算与数组维度变换
from torch.utils.data import DataLoader  # PyTorch 数据加载器
from lerobot.datasets.utils import serialize_dict  # 将统计量字典中的张量/数组序列化为 JSON 兼容格式
from lerobot.utils.io_utils import write_json  # 将字典写入 JSON 文件的工具函数
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # LeRobot 数据集读取类
from mujoco_env.y_env import SimpleEnv  # 自定义 MuJoCo 桌面抓取仿真环境

# if youu want to use the example data provided, DATAPATH = './demo_data_example' instead!
DATAPATH = './data/demo_data_20261118_085434'  # 待可视化与回放的数据集目录路径

# Select an episode index that you want to visualize
episode_index = 0  # 指定要回放查看的 episode 索引

dataset = LeRobotDataset('omy_pnp', root=DATAPATH, episodes=[episode_index])  # 仅加载指定 episode 的数据集切片
dataloader = DataLoader(dataset, batch_size=1, num_workers=0)  # 按顺序逐帧读取（batch_size=1，单进程）

xml_path = './asset/example_scene_y.xml'  # MuJoCo 场景 XML 文件路径
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')  # 初始化仿真环境，动作模式设为绝对关节角 'joint_angle'

step = 0  # 当前回放帧索引计数器
iter_dataloader = iter(dataloader)  # 创建 DataLoader 迭代器
PnPEnv.reset()  # 重置仿真环境到初始状态

while PnPEnv.env.is_viewer_alive():  # 只要 MuJoCo 窗口保持打开，就循环回放
    PnPEnv.step_env()  # 推进一步 MuJoCo 物理仿真
    if PnPEnv.env.loop_every(HZ=20):  # 以 20Hz 的频率从数据集中读取并回放帧
        # Get the action from dataset
        data = next(iter_dataloader)  # 取出当前时间步的一帧数据
        if step == 0:  # 在第 0 帧时同步物体初始位置
            # Reset the object pose based on the dataset
            PnPEnv.set_obj_pose(
                data['obj_init'][0, :3].numpy(),  # 设置杯子的初始 [x, y, z] 位置
                data['obj_init'][0, 3:].numpy()  # 设置盘子的初始 [x, y, z] 位置
            )
        # Get the action from dataset
        action = data['action'].numpy()  # 提取该帧记录的动作（形状为 (1, 7)）
        PnPEnv.step(action[0])  # 将 7 维关节角与夹爪指令应用到仿真机械臂

        # Visualize the image from dataset to rgb_overlay
        PnPEnv.rgb_agent = (data['observation.image'][0].numpy() * 255).astype(np.uint8)  # 将归一化的主视角图像反缩放回 [0, 255] uint8
        PnPEnv.rgb_ego = (data['observation.wrist_image'][0].numpy() * 255).astype(np.uint8)  # 将归一化的腕部图像反缩放回 [0, 255] uint8
        # 3 256 256 -> 256 256 3
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1,2,0))  # 通道维从 CHW 转为 HWC 以便渲染显示
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1,2,0))  # 通道维从 CHW 转为 HWC 以便渲染显示
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)  # 侧视角占位黑图（回放模式不显示侧视角）
        PnPEnv.render()  # 渲染仿真画面与数据集图像叠加层
        step += 1  # 步数加 1

        if step == len(dataset):  # 若当前 episode 已回放完毕
            # start from the beginning
            iter_dataloader = iter(dataloader)  # 重置迭代器以循环播放
            PnPEnv.reset()  # 重置仿真环境
            step = 0  # 重置帧计数器

PnPEnv.env.close_viewer()  # 关闭窗口后清理 MuJoCo viewer 资源

stats = dataset.meta.stats  # 读取数据集元数据中的统计信息（均值、方差、最值等）
PATH = dataset.root / 'meta' / 'stats.json'  # 目标 stats.json 文件路径
stats = serialize_dict(stats)  # 将统计字典序列化为可写入 JSON 的标准 Python 类型

write_json(stats, PATH)  # 将统计信息导出保存为 stats.json
