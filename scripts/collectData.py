from datetime import datetime  # 导入日期时间模块，用于按当前时间生成唯一的数据集目录名
import lerobot  # 导入 LeRobot 库，用于打印并核对当前安装的框架版本
import numpy as np  # 导入 NumPy 数值计算库，用于处理状态、动作与图像数组
from PIL import Image  # 导入 Pillow 图像库，用于将仿真相机画面缩放到 256×256
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 导入 LeRobot 数据集创建、追加与收尾保存类

from mujoco_env.y_env import SimpleEnv  # 导入多物体语言条件跨障分拣 MuJoCo 仿真环境

# ==================== 采集配置区 ====================
SEED = None  # 随机种子：None 表示每条轨迹随机生成物体坐标（训练推荐）；填整数（如 0）可固定初位调试
REPO_NAME = "omy_sorting_vla"  # 数据集仓库标识名（需与后续训练配置中的 repo_id 保持一致）
NUM_COLLECT = 20  # 本次运行计划采集的成功演示轨迹（episode）总条数
XML_PATH = "./asset/example_scene_y.xml"  # MuJoCo 跨障分拣场景主 XML 文件路径

# 任务指令调度模式配置：
#   "cycle"       : 按 0→1→2→3 循环轮询 4 条语言指令（推荐！每条指令样本量严格均衡，适合 SmolVLA / Pi0）
#   "random"      : 每次重置从 4 条指令中均匀随机抽取
#   0 / 1 / 2 / 3 : 固定只采集某一条单任务指令（适合先快速验证单任务 ACT baseline）
#       0 -> Pick up the apple and place it in the blue tray.
#       1 -> Pick up the apple and place it in the green tray.
#       2 -> Pick up the orange and place it in the blue tray.
#       3 -> Pick up the orange and place it in the green tray.
# TASK_MODE = "cycle"  # 默认采用 4 指令均衡轮询模式
TASK_MODE = 0

# OOD（分布外）开关配置：采集标准训练集时保持 False；做数据增强或采集 OOD 测试样例时可开启
OOD_DISTRACTOR = False  # 是否在左侧抓取区额外放入红牛罐干扰物
OOD_POSITION = False  # 是否扩大抓取区物体位置随机采样范围

CREATE_NEW_DATASET = True  # True=创建全新数据集目录；False=在 OLD_DATASET_PATH 上断点续录追加轨迹
OLD_DATASET_PATH = "./data/sorting_data_20261119_000000"  # 仅当 CREATE_NEW_DATASET=False 时生效的旧数据集路径
# ====================================================


def get_next_task_idx(ep_idx, mode):
    """根据当前已采集的全局轨迹序号 ep_idx 与调度模式 mode，决定下一条 episode 的任务索引"""
    if isinstance(mode, int):  # 若配置为固定整数（0/1/2/3）
        return mode  # 始终返回该固定任务索引
    if mode == "cycle":  # 若配置为轮询模式
        return ep_idx % len(SimpleEnv.TASKS)  # 对 4 取模实现四条指令均匀交替出现
    return None  # 其他模式（如 "random"）返回 None，交由 SimpleEnv.reset() 随机抽取


ROOT = f"./data/sorting_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # 拼接带时间戳的新数据集本地保存路径
print(f"lerobot version: {lerobot.__version__}")  # 打印当前环境中的 LeRobot 版本号

if CREATE_NEW_DATASET:  # 分支 1：创建全新的 LeRobot 数据集
    print(f"Create new dataset: {ROOT}")  # 打印新建数据集根目录
    dataset = LeRobotDataset.create(  # 初始化新数据集并声明各模态特征 Schema
        repo_id=REPO_NAME,  # 数据集名称标识
        root=ROOT,  # 本地磁盘保存目录
        robot_type="omy",  # 机器人本体型号标识
        fps=20,  # 采样与控制帧率：20Hz（每秒写入 20 帧）
        features={  # 定义每帧需要记录的数据字段、类型与形状
            "observation.image": {  # 第三人称固定主视角 RGB 图像
                "dtype": "image",  # 图像模态（底层自动按 LeRobot 图像/视频格式存储）
                "shape": (256, 256, 3),  # 高 256、宽 256、RGB 3 通道
                "names": ["height", "width", "channels"],  # 各维度语义名称
            },
            "observation.wrist_image": {  # 机械臂腕部第一人称视角 RGB 图像
                "dtype": "image",  # 图像模态
                "shape": (256, 256, 3),  # 高 256、宽 256、RGB 3 通道
                "names": ["height", "width", "channels"],  # 各维度语义名称
            },
            "observation.state": {  # 机器人本体状态观测：6 维臂关节角（与 ACT / SmolVLA / Pi0 全部统一）
                "dtype": "float32",  # 32 位单精度浮点
                "shape": (6,),  # 长度 6：[joint1..joint6]
                "names": ["state"],  # 维度名称
            },
            "action": {  # 监督动作标签：执行 IK 后的 6 维目标关节角 + 1 维夹爪开合指令
                "dtype": "float32",  # 32 位单精度浮点
                "shape": (7,),  # 长度 7：[j1..j6, gripper]
                "names": ["action"],  # 维度名称
            },
            "obj_init": {  # 物体初始世界坐标（苹果 xyz + 橙子 xyz + 红牛罐 xyz），仅用于精确回放，不参与策略训练
                "dtype": "float32",  # 32 位单精度浮点
                "shape": (9,),  # 长度 9：3 个物体 × 3 维坐标
                "names": ["obj_init"],  # 维度名称
            },
        },
    )
else:  # 分支 2：加载已有数据集并继续追加录制
    print(f"Resume from previous dataset: {OLD_DATASET_PATH}")  # 打印续录目录路径
    dataset = LeRobotDataset.resume(  # 打开已有数据集元信息并准备追加
        repo_id=REPO_NAME,  # 数据集名称标识
        root=OLD_DATASET_PATH,  # 旧数据集本地根目录
    )
    print(f"Existing episodes: {dataset.meta.total_episodes}")  # 打印当前已包含的轨迹条数

start_ep_offset = 0 if CREATE_NEW_DATASET else dataset.meta.total_episodes  # 计算续录起始偏移量，保证轮询任务衔接不断档
cur_task_idx = get_next_task_idx(start_ep_offset, TASK_MODE)  # 计算第 1 条待采集轨迹对应的任务索引

PnPEnv = SimpleEnv(  # 实例化跨障分拣 MuJoCo 仿真环境并弹出可视化窗口
    XML_PATH,  # 场景 XML 路径
    action_type="eef_pose",  # 采集时采用末端位姿增量模式，方便键盘与手柄直觉操作
    state_type="joint_angle",  # 状态空间设为关节角
    seed=SEED,  # 随机种子
    task_idx=cur_task_idx,  # 初始任务索引
    ood_distractor=OOD_DISTRACTOR,  # 是否开启红牛罐干扰物
    ood_position=OOD_POSITION,  # 是否开启 OOD 位置扩展
)

action = np.zeros(7, dtype=np.float32)  # 初始化 7 维零动作缓存
episode_id = 0  # 本次运行已成功保存的轨迹计数器
record_flag = False  # 录制触发标志：等待首次有效操控输入后再置 True，避免录入开头静止发呆帧

try:  # 使用 try...finally 确保中途关闭窗口或 Ctrl+C 时也能安全收尾保存已录完的数据
    while PnPEnv.env.is_viewer_alive() and episode_id < NUM_COLLECT:  # 只要窗口未关闭且未采够目标条数就持续循环
        PnPEnv.step_env()  # 以 500Hz 推进一步 MuJoCo 底层物理仿真
        if PnPEnv.env.loop_every(HZ=20):  # 每隔 25 个物理步（即 20Hz）执行一次遥操作控制与数据采样
            done = PnPEnv.check_success()  # 检查当前帧是否已满足全部任务成功条件
            if done:  # 若本条演示已成功完成
                if dataset.has_pending_frames():  # 确认缓冲区中确实已有录制帧（防止空轨迹误存）
                    stage = PnPEnv.get_stage_info()  # 读取本条轨迹的抓起/越障/入框分阶段完成信息
                    dataset.save_episode()  # 将当前 episode 写入磁盘并更新数据集统计量
                    episode_id += 1  # 已完成轨迹数加 1
                    print(  # 打印保存成功日志、对应指令与三阶段达成状态
                        f"Saved episode {episode_id}/{NUM_COLLECT} "
                        f"| Task: {PnPEnv.instruction} "
                        f"| Stages: grasp={stage['grasped']}, cross={stage['crossed_barrier']}, place={stage['placed']}"
                    )
                else:  # 尚未开始移动就意外满足成功条件时跳过保存
                    print("Success detected before any frame was recorded, skip saving.")

                next_task_idx = get_next_task_idx(start_ep_offset + episode_id, TASK_MODE)  # 计算下一条轨迹的任务索引
                PnPEnv.reset(  # 重置机械臂、随机新物体位置并切换到下一条指令
                    seed=SEED,  # 随机种子
                    task_idx=next_task_idx,  # 下一条任务索引
                    ood_distractor=OOD_DISTRACTOR,  # 保持 OOD 干扰物设置
                    ood_position=OOD_POSITION,  # 保持 OOD 位置设置
                )
                record_flag = False  # 重置录制触发标志，等待用户开始下一次操作
                action = np.zeros(7, dtype=np.float32)  # 清零动作缓存
            else:  # 若本条任务尚未完成，继续读取遥操作输入并推进动作
                teleop_action, reset = PnPEnv.teleop_robot()  # 读取键盘 + 手柄叠加输入，返回末端增量动作与重置信号

                if not record_flag and np.any(np.abs(teleop_action) > 1e-6):  # 检测到首次有效遥控动作时
                    record_flag = True  # 正式开启本条 episode 录帧
                    print(f"Start recording ep {episode_id + 1}/{NUM_COLLECT} | {PnPEnv.instruction}")  # 打印开录提示

                if reset:  # 若用户按下键盘 Z 或手柄 Y/Start 请求放弃重来
                    PnPEnv.reset(  # 重置仿真环境并保持同一条任务指令重做（不跳题）
                        seed=SEED,
                        task_idx=get_next_task_idx(start_ep_offset + episode_id, TASK_MODE),
                        ood_distractor=OOD_DISTRACTOR,
                        ood_position=OOD_POSITION,
                    )
                    dataset.clear_episode_buffer()  # 丢弃当前未完成的缓存帧
                    record_flag = False  # 关闭录制标志，等待下一次移动重新开录
                    action = np.zeros(7, dtype=np.float32)  # 清零动作缓存
                    continue  # 跳过本帧写入，直接进入下一循环

                # 1) 在执行本帧动作前采集观测（严格遵循“当前观测 o_t → 执行动作 a_t”的模仿学习时序对齐）
                obs_joint_state = np.asarray(PnPEnv.get_joint_state()[:6], dtype=np.float32)  # 读取 6 维当前真实关节角
                agent_image, wrist_image = PnPEnv.grab_image()  # 抓取主视角与腕部视角原始 RGB 图像
                agent_image = np.asarray(  # 将主视角图像缩放至 256×256 并转为 uint8 数组
                    Image.fromarray(agent_image).resize((256, 256)), dtype=np.uint8
                )
                wrist_image = np.asarray(  # 将腕部视角图像缩放至 256×256 并转为 uint8 数组
                    Image.fromarray(wrist_image).resize((256, 256)), dtype=np.uint8
                )

                # 2) 将末端增量指令送入环境做 IK 求解，提取目标 6 轴关节角 + 1 维夹爪指令作为动作标签
                PnPEnv.step(teleop_action)  # 内部更新 self.q（10 维底层目标：前 6 维臂关节 + 第 7 维夹爪主关节）
                target_action = np.asarray(PnPEnv.q[:7], dtype=np.float32)  # 切片取 7 维动作标签存入数据集

                if record_flag:  # 若已处于录制状态，将本帧完整多模态数据写入缓冲区
                    dataset.add_frame(
                        {
                            "observation.image": agent_image,  # 主视角图像 o_t
                            "observation.wrist_image": wrist_image,  # 腕部视角图像 o_t
                            "observation.state": obs_joint_state,  # 6 维关节状态 o_t
                            "action": target_action,  # 7 维目标关节+夹爪动作 a_t
                            "obj_init": np.asarray(PnPEnv.obj_init_pose, dtype=np.float32),  # 9 维物体初位
                            "task": PnPEnv.instruction,  # 当前自然语言任务描述
                        }
                    )

            PnPEnv.render(teleop=True, idx=episode_id)  # 刷新渲染窗口并叠加三视角、任务指令与手柄状态
finally:  # 无论正常采完还是中途关闭窗口，都会执行以下清理与落盘逻辑
    if dataset.has_pending_frames():  # 若退出时尚有未达到成功条件的残余缓冲帧
        print("Discard unfinished episode.")  # 提示丢弃残缺轨迹，防止污染训练集
        dataset.clear_episode_buffer()  # 清空未保存缓冲区
    dataset.finalize()  # 执行 LeRobot 数据集收尾（刷新 Parquet、编码视频并写入完整元信息）
    PnPEnv.env.close_viewer()  # 关闭并释放 MuJoCo 渲染窗口资源

print(f"Dataset saved to: {dataset.root}")  # 打印最终数据集落盘目录路径
