import numpy as np  # 导入 NumPy 数值计算库，用于图像维度转置与数组转换
from torch.utils.data import DataLoader  # 导入 PyTorch DataLoader 按帧顺序读取数据集
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 导入 LeRobot 数据集加载类
from lerobot.datasets.utils import serialize_dict  # 导入统计字典序列化工具，将张量转为 JSON 可写格式
from lerobot.utils.io_utils import write_json  # 导入 JSON 写入工具

from mujoco_env.y_env import SimpleEnv  # 导入多物体语言条件跨障分拣仿真环境

DATAPATH = "./data/sorting_data_20261119_000000"  # 待回放质检的本地数据集目录（采集后替换为实际路径）
REPO_NAME = "omy_sorting_vla"  # 数据集名称标识（需与 collectData.py 中 REPO_NAME 保持一致）
EPISODE_INDEX = 0  # 指定要回放查看的 episode 编号（从 0 开始）

dataset = LeRobotDataset(REPO_NAME, root=DATAPATH, episodes=[EPISODE_INDEX])  # 仅加载指定编号的单条轨迹
dataloader = DataLoader(dataset, batch_size=1, num_workers=0)  # 单帧单进程顺序读取，保证严格按时间步回放

xml_path = "./asset/example_scene_y.xml"  # MuJoCo 跨障分拣场景 XML 文件路径
PnPEnv = SimpleEnv(xml_path, action_type="joint_angle")  # 以绝对关节角模式初始化仿真环境用于动作回放

step = 0  # 当前已回放的帧号计数器
iter_dataloader = iter(dataloader)  # 获取 DataLoader 的初始迭代器
PnPEnv.reset()  # 首次重置仿真环境

while PnPEnv.env.is_viewer_alive():  # 只要 MuJoCo 可视化窗口保持打开就持续循环回放
    PnPEnv.step_env()  # 推进一步底层物理仿真（500Hz）
    if PnPEnv.env.loop_every(HZ=20):  # 按与采集一致的 20Hz 频率逐帧回放动作
        data = next(iter_dataloader)  # 读取当前时间步的数据字典
        if step == 0:  # 在每轮回放的第 0 帧恢复录制时的物体初位与语言指令
            obj_init = data["obj_init"][0].numpy()  # 提取 9 维物体初始坐标向量
            if len(obj_init) >= 9:  # 新版跨障分拣数据集：含苹果、橙子、红牛罐共 9 维坐标
                PnPEnv.set_obj_pose(obj_init[:3], obj_init[3:6], obj_init[6:9])  # 精确还原 3 个物体初位
            else:  # 兼容仅含前两个物体坐标的 6 维旧格式
                PnPEnv.set_obj_pose(obj_init[:3], obj_init[3:6])  # 仅还原苹果与橙子，红牛罐停放场外
            if "task" in data:  # 若数据帧中包含自然语言任务描述
                # DataLoader batch=1 时字符串会被包装成长度为 1 的列表/元组，这里解包为纯字符串
                task_str = (
                    data["task"][0]
                    if isinstance(data["task"], (list, tuple))
                    else str(data["task"])
                )
                PnPEnv.set_instruction(given=task_str)  # 将当前任务指令同步显示到渲染窗口

        action = data["action"].numpy()  # 读取本帧记录的 7 维关节+夹爪动作（形状 (1, 7)）
        PnPEnv.step(action[0])  # 将动作指令下发到仿真机械臂复现动作轨迹

        # 提取数据集中实际保存的图像（浮点 [0,1] CHW）并转回 uint8 HWC，叠加到右上/右下小窗核对画质
        PnPEnv.rgb_agent = (data["observation.image"][0].numpy() * 255).astype(np.uint8)  # 还原主视角像素值到 [0, 255]
        PnPEnv.rgb_ego = (data["observation.wrist_image"][0].numpy() * 255).astype(np.uint8)  # 还原腕部视角像素值
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1, 2, 0))  # 通道维从 (3, 256, 256) 转为 (256, 256, 3)
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1, 2, 0))  # 通道维从 (3, 256, 256) 转为 (256, 256, 3)
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)  # 回放模式下侧视角用全黑占位图即可
        PnPEnv.render(idx=EPISODE_INDEX)  # 刷新窗口画面并标注当前回放的 Episode 编号与任务指令
        step += 1  # 帧号加 1

        if step == len(dataset):  # 若当前 episode 已播到最后一帧
            iter_dataloader = iter(dataloader)  # 重建迭代器从头循环播放
            PnPEnv.reset()  # 重置机械臂与仿真环境
            step = 0  # 帧号归零

PnPEnv.env.close_viewer()  # 用户关闭窗口后释放 MuJoCo 渲染器资源
