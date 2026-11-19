import numpy as np  # 数值计算库
from datetime import datetime  # 用于生成带时间戳的数据集保存目录名
from PIL import Image  # 用于调整相机图像分辨率
import lerobot  # 导入 LeRobot 库以读取版本号
from mujoco_env.y_env import SimpleEnv  # 导入自定义的 MuJoCo 桌面抓取仿真环境
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # LeRobot 数据集读写类

SEED = 0  # 随机种子，固定物体初始位置便于复现
# SEED = None <- Uncomment this line to randomize the object positions  # 取消注释可每次随机化物体位置
REPO_NAME = "omy_pnp"  # 数据集仓库名称标识
NUM_DEMO = 1  # 需要采集的演示轨迹（episode）总条数
TASK_NAME = "Put mug cup on the plate"  # 当前采集任务的自然语言描述
XML_PATH = "./asset/example_scene_y.xml"  # MuJoCo 场景模型 XML 文件路径
CREATE_NEW_DATASET = (
    True  # True 表示新建数据集；False 表示在已有数据集上继续追加录制
)
OLD_DATASET_PATH = (
    "./data/demo_data_20261117_161335"  # 仅当 CREATE_NEW_DATASET = False 时使用的旧数据集路径
)

ROOT = f"./data/demo_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # 按当前时间生成新数据集的保存目录
print(f"lerobot version: {lerobot.__version__}")  # 打印当前安装的 LeRobot 版本

if CREATE_NEW_DATASET:  # 若选择创建新数据集
    print(f"Create new dataset: {ROOT}")  # 打印新建数据集路径
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,  # 数据集名称
        root=ROOT,  # 本地保存根目录
        robot_type="omy",  # 机器人类型标识
        fps=20,  # 采集帧率：每秒 20 帧
        features={
            "observation.image": {  # 主视角（第三人称）RGB 图像特征
                "dtype": "image",  # 图像类型（自动按视频/图像格式编码存储）
                "shape": (256, 256, 3),  # 图像尺寸（高、宽、通道数）
                "names": ["height", "width", "channels"],  # 各维度名称
            },
            "observation.wrist_image": {  # 腕部视角（第一人称）RGB 图像特征
                "dtype": "image",  # 图像类型
                "shape": (256, 256, 3),  # 图像尺寸（高、宽、通道数）
                "names": ["height", "width", "channels"],  # 各维度名称
            },
            "observation.state": {  # 机器人末端位姿状态
                "dtype": "float32",  # 32 位浮点型
                "shape": (6,),  # 6 维向量
                "names": ["state"],  # x, y, z 位置与 roll, pitch, yaw 姿态角
            },
            "action": {  # 动作标签（记录实际执行的关节角与夹爪状态）
                "dtype": "float32",  # 32 位浮点型
                "shape": (7,),  # 7 维向量
                "names": ["action"],  # 6 个机械臂关节角 + 1 个夹爪状态
            },
            "obj_init": {  # 物体初始位置信息（用于复现/回放，不参与策略训练）
                "dtype": "float32",  # 32 位浮点型
                "shape": (6,),  # 6 维向量（杯子 xyz + 盘子 xyz）
                "names": ["obj_init"],  # initial object pose, not used in training
            },
        }
    )
else:  # 若选择在已有数据集上断点续录
    print(f"Resume from previous dataset: {OLD_DATASET_PATH}")  # 打印加载的旧数据集路径
    dataset = LeRobotDataset.resume(
        repo_id=REPO_NAME,  # 数据集名称
        root=OLD_DATASET_PATH,  # 旧数据集所在根目录
    )
    print(f"Existing episodes: {dataset.meta.total_episodes}")  # 打印已有数据集中的 episode 数量

# Define the environment
PnPEnv = SimpleEnv(XML_PATH, seed=SEED)  # 初始化 MuJoCo 仿真环境（默认 action_type="eef_pose" 用于键盘遥操作）
action = np.zeros(7)  # 初始化 7 维零动作（dx, dy, dz, droll, dpitch, dyaw, gripper）
episode_id = 0  # 当前已成功保存的 episode 计数
record_flag = False  # 录制标志位：当机械臂开始移动时再置为 True，避免录入静止等待帧

try:
    while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:  # 只要可视化窗口未关闭且未采够数量就持续循环
        PnPEnv.step_env()  # 推进一步 MuJoCo 物理仿真（默认控制频率）
        if PnPEnv.env.loop_every(HZ=20):  # 以 20Hz 的控制与采样频率执行以下逻辑
            done = PnPEnv.check_success()  # 检查当前是否满足任务成功条件（杯子直立静止在盘中且夹爪松开抬起）
            if done:  # 若当前演示成功完成
                if dataset.has_pending_frames():  # 若当前缓冲区中有待保存的帧数据
                    dataset.save_episode()  # 将当前 episode 写入磁盘并更新统计信息
                    episode_id += 1  # 已完成轨迹数加 1
                    print(f"Saved episode {episode_id}/{NUM_DEMO} to {dataset.root}")  # 打印保存成功提示
                else:  # 若未录制任何帧就触发了成功判定
                    print(
                        "Success detected before any frame was recorded, skip saving."  # 跳过保存空轨迹
                    )
                PnPEnv.reset(seed=SEED)  # 重置仿真环境以准备下一条演示
                record_flag = False  # 重置录制标志，等待下一次移动再开始录制
                action = np.zeros(7)  # 清零动作指令
            else:  # 若任务尚未完成，继续遥操作控制
                action, reset = PnPEnv.teleop_robot()  # 读取键盘输入，返回增量末端动作与是否按下重置键 'z'

                if not record_flag and np.any(np.abs(action) > 1e-6):  # 检测到首次有效键盘控制输入时
                    record_flag = True  # 开启录制
                    print("Start recording")  # 打印开始录制提示
                    
                if reset:  # 若用户按下了 'z' 键请求重置当前轨迹
                    # Press 'z' to reset the environment and clear the current episode buffer
                    PnPEnv.reset(seed=SEED)  # 重置仿真环境
                    dataset.clear_episode_buffer()  # 清空当前未保存的 episode 缓存帧
                    record_flag = False  # 关闭录制标志
                    action = np.zeros(7)  # 清零动作指令

                ee_pose = np.asarray(PnPEnv.get_ee_pose(), dtype=np.float32)  # 获取当前末端执行器 6D 位姿（x, y, z, r, p, y）
                agent_image, wrist_image = PnPEnv.grab_image()  # 从仿真相机抓取主视角与腕部视角图像
                agent_image = np.asarray(
                    Image.fromarray(agent_image).resize((256, 256)), dtype=np.uint8  # 将主视角图像缩放到 256x256 uint8
                )
                wrist_image = np.asarray(
                    Image.fromarray(wrist_image).resize((256, 256)), dtype=np.uint8  # 将腕部图像缩放到 256x256 uint8
                )
                joint_q = np.asarray(PnPEnv.step(action), dtype=np.float32)  # 执行末端增量动作（内部做 IK 求解），返回执行后的 7 维关节+夹爪状态

                if record_flag:  # 若已处于录制状态，则将该帧数据写入缓冲区
                    dataset.add_frame(
                        {
                            "observation.image": agent_image,  # 主视角图像
                            "observation.wrist_image": wrist_image,  # 腕部视角图像
                            "observation.state": ee_pose,  # 末端位姿观测
                            "action": joint_q,  # 目标关节角 + 夹爪状态作为动作标签
                            "obj_init": np.asarray(
                                PnPEnv.obj_init_pose, dtype=np.float32  # 物体初始位置（杯子与盘子坐标）
                            ),
                            "task": TASK_NAME,  # 任务描述文本
                        }
                    )
            PnPEnv.render(teleop=True)  # 刷新 MuJoCo 渲染窗口（含侧视角与按键提示叠加层）
finally:
    if dataset.has_pending_frames():  # 退出前若仍有未保存的残余帧（例如中途关闭窗口）
        print("Discard unfinished episode.")  # 提示丢弃未完成的轨迹
        dataset.clear_episode_buffer()  # 清空未保存的缓冲区
    dataset.finalize()  # 完成数据集收尾（刷新 parquet / 视频编码与元数据写入）
    PnPEnv.env.close_viewer()  # 关闭 MuJoCo 可视化窗口

print(f"Dataset saved to: {dataset.root}")  # 打印最终数据集保存路径
