import time  # 用于统计单帧策略推理耗时与 FPS（便于简历横向对比 ACT / SmolVLA / Pi0）
import numpy as np  # 数值计算库，用于处理观测状态与输出动作数组
from PIL import Image  # 图像缩放工具，将仿真相机画面统一调整到 256×256
import torch  # PyTorch 深度学习推理框架
import torchvision  # 提供 ToTensor() 图像张量转换（HWC uint8 [0,255] -> CHW float32 [0,1]）

from lerobot.policies.act.configuration_act import ACTConfig  # ACT 策略配置类
from lerobot.policies.act.modeling_act import ACTPolicy  # ACT 策略网络模型类
from lerobot.policies.factory import make_pre_post_processors  # LeRobot 0.7.0 预处理/后处理器工厂函数

from mujoco_env.y_env import SimpleEnv  # 多物体语言条件跨障分拣 MuJoCo 仿真环境

# ==================== 部署与评测配置区 ====================
CKPT_PATH = "./ckpt/act_y"  # 待部署的本地 ACT 权重目录（内含 model.safetensors 与 pre/post processor）
XML_PATH = "./asset/example_scene_y.xml"  # MuJoCo 场景模型 XML 文件路径

NUM_EPISODES = 10  # 连续评测的回合（episode）总数
MAX_STEPS = 400  # 单个回合最大控制步数（20Hz 下 400 步 = 20 秒），超时判失败并进入下一回合
SEED = None  # 随机种子：None 表示每回合随机生成物体位置；填整数（如 0）可固定初位复现

# 任务与 OOD 评测配置（与 SimpleEnv.TASKS 对应）：
#   0: Pick up the apple and place it in the blue tray.
#   1: Pick up the apple and place it in the green tray.
#   2: Pick up the orange and place it in the blue tray.
#   3: Pick up the orange and place it in the green tray.
#   None: 每回合随机抽取 4 条指令之一
TASK_IDX = 0  # 默认评测任务索引（若训练的是单任务 ACT，请改为与训练集一致的任务号）
INSTRUCTION_OVERRIDE = None  # 可选：自定义同义改写指令，不为 None 时覆盖 TASK_IDX
OOD_DISTRACTOR = False  # True 时在抓取区加入红牛罐干扰物（测试抗干扰泛化）
OOD_POSITION = False  # True 时扩大物体位置采样范围（测试空间外推泛化）

# 状态输入类型：须与训练数据集 observation.state 保持一致
#   "joint_angle": 6 维关节角（对应当前 scripts/collectData.py 新场景采集的数据）
#   "ee_pose"    : 6 维末端位姿（仅当权重是用旧版 demo_data_example 训练时使用）
STATE_TYPE = "joint_angle"

# 时序集成（Temporal Ensemble）：每步重推理并加权平滑历史动作块，轨迹更顺滑
# 若希望直接按训练时的 n_action_steps（如每 10 步推理一次）运行，将其设为 False
USE_TEMPORAL_ENSEMBLE = True
TEMPORAL_ENSEMBLE_COEFF = 0.9  # 指数加权系数（仅当 USE_TEMPORAL_ENSEMBLE=True 时生效）
# ========================================================


def load_act_policy(ckpt_path: str, device: str):
    """离线加载本地 ACT 模型权重及其配套的输入预处理器与输出后处理器。"""
    cfg = ACTConfig.from_pretrained(ckpt_path)  # 从本地 config.json 读取训练时的网络结构与特征配置
    cfg.device = device  # 将推理设备更新为当前可用的 cuda / cpu
    if USE_TEMPORAL_ENSEMBLE:  # 若启用时序集成平滑输出
        cfg.n_action_steps = 1  # 时序集成要求每个控制步执行 1 步并重新融合预测块
        cfg.temporal_ensemble_coeff = TEMPORAL_ENSEMBLE_COEFF  # 设置时序指数加权系数

    policy = ACTPolicy.from_pretrained(ckpt_path, config=cfg)  # 从本地权重目录加载 ACT 参数
    policy.to(device)  # 将模型参数与缓冲区迁移到目标设备
    policy.eval()  # 切换为评估模式（关闭 Dropout，推理时不使用 CVAE 编码器分支）

    # 直接从权重目录加载训练时保存的归一化/反归一化处理器，无需再读取原始数据集元信息
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,  # 传入策略配置
        pretrained_path=ckpt_path,  # 指定本地保存的 processor 配置与 safetensors 统计量路径
        preprocessor_overrides={"device_processor": {"device": device}},  # 确保输入数据自动迁移到当前设备
    )
    return policy, preprocessor, postprocessor  # 返回模型与前后处理器三元组


def build_observation(env: SimpleEnv, policy: ACTPolicy, img_transform, device: str):
    """采集当前仿真状态与相机画面，组装为 LeRobot 预处理器所需的观测字典。"""
    if STATE_TYPE == "joint_angle":  # 读取 6 维机械臂当前关节角（与新场景采集格式一致）
        state = env.get_joint_state()[:6]
    elif STATE_TYPE == "ee_pose":  # 读取 6 维末端执行器位姿（兼容旧版数据集格式）
        state = env.get_ee_pose()
    else:
        raise ValueError(f"Unsupported STATE_TYPE: {STATE_TYPE}")

    agent_img, wrist_img = env.grab_image()  # 从 MuJoCo 固定相机抓取主视角与腕部视角 RGB 图像
    agent_tensor = img_transform(Image.fromarray(agent_img).resize((256, 256)))  # 缩放至 256×256 并转为 CHW [0,1] 张量

    data = {  # 构造批次维度为 1 的观测字典
        "observation.state": torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),  # 形状 (1, 6)
        "observation.image": agent_tensor.unsqueeze(0).to(device),  # 形状 (1, 3, 256, 256)
    }

    # 若当前权重训练时也使用了腕部相机，则自动附带 observation.wrist_image
    if "observation.wrist_image" in policy.config.input_features:
        wrist_tensor = img_transform(Image.fromarray(wrist_img).resize((256, 256)))  # 缩放并转张量
        data["observation.wrist_image"] = wrist_tensor.unsqueeze(0).to(device)  # 形状 (1, 3, 256, 256)

    return data  # 返回待预处理的多模态观测字典


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 优先使用 GPU 推理，无 GPU 时回退 CPU
    print(f"Device: {device}")
    print(f"Loading ACT checkpoint from: {CKPT_PATH}")

    policy, preprocessor, postprocessor = load_act_policy(CKPT_PATH, device)  # 加载策略与处理器
    print(f"Policy input features: {list(policy.config.input_features.keys())}")  # 打印策略实际使用的输入模态

    # 初始化 MuJoCo 仿真环境，动作模式设为绝对关节角 'joint_angle'（与 ACT 输出的 7 维动作一致）
    env = SimpleEnv(
        XML_PATH,
        action_type="joint_angle",
        state_type="joint_angle",
        seed=SEED,
        task_idx=TASK_IDX,
        ood_distractor=OOD_DISTRACTOR,
        ood_position=OOD_POSITION,
    )

    img_transform = torchvision.transforms.ToTensor()  # 图像数组转浮点张量工具
    results = []  # 记录每个评测回合的分阶段结果与耗时
    infer_times_ms = []  # 记录单步推理耗时（毫秒），用于统计平均延迟与 FPS

    try:
        for ep in range(NUM_EPISODES):  # 依次运行 NUM_EPISODES 个评测回合
            if not env.env.is_viewer_alive():  # 若用户提前关闭可视化窗口则中止评测
                print("Viewer closed by user, stopping evaluation.")
                break

            ep_seed = None if SEED is None else SEED + ep  # 固定种子时每回合递增种子，便于复现不同初始布局
            env.reset(  # 重置机械臂、随机物体初位与语言指令
                seed=ep_seed,
                task_idx=None if INSTRUCTION_OVERRIDE else TASK_IDX,
                instruction=INSTRUCTION_OVERRIDE,
                ood_distractor=OOD_DISTRACTOR,
                ood_position=OOD_POSITION,
            )
            policy.reset()  # 清空 ACT 内部缓存的历史动作块与时序集成队列

            step = 0  # 当前回合控制步数计数器
            success = False  # 本回合是否达成任务成功条件

            while env.env.is_viewer_alive() and step < MAX_STEPS:  # 在步数上限内以 20Hz 闭环推理控制
                env.step_env()  # 推进一步底层 MuJoCo 物理仿真（500Hz）
                if env.env.loop_every(HZ=20):  # 每 25 个物理步（20Hz）执行一次策略推理与动作下发
                    data = build_observation(env, policy, img_transform, device)  # 采集当前观测

                    t0 = time.perf_counter()  # 记录推理起始时间戳
                    with torch.inference_mode():  # 关闭梯度计算以加速推理并降低显存占用
                        batch = preprocessor(data)  # 对状态与图像做归一化并送入设备
                        action = policy.select_action(batch)  # 调用 ACT 输出当前步（归一化）动作
                        action = postprocessor(action)  # 反归一化还原真实关节角与夹爪指令尺度
                    infer_times_ms.append((time.perf_counter() - t0) * 1000.0)  # 累计单步耗时（ms）

                    action_np = action[0].cpu().numpy()  # 转为形状 (7,) 的 NumPy 动作向量
                    env.step(action_np)  # 下发 6 轴目标关节角 + 1 维夹爪指令到仿真环境
                    env.render(idx=ep)  # 刷新渲染窗口并显示当前 Episode 编号与任务指令
                    step += 1  # 控制步数加 1

                    if env.check_success():  # 检查是否已完成全部任务条件（入盘、静止、松爪、抬手）
                        success = True  # 标记本回合成功
                        break  # 提前结束本回合进入下一回合

            stage = env.get_stage_info()  # 读取本回合的抓起/越障/入框分阶段统计
            stage["success"] = success  # 写入最终成功标志
            stage["steps"] = step  # 记录耗时步数
            results.append(stage)  # 存入汇总列表
            print(
                f"[Episode {ep + 1}/{NUM_EPISODES}] "
                f"success={success} | steps={step} "
                f"| grasp={stage['grasped']}, cross={stage['crossed_barrier']}, place={stage['placed']}"
            )
    finally:
        env.env.close_viewer()  # 无论正常完成还是中断退出，均释放 MuJoCo 渲染窗口

    # 打印多回合汇总指标（可直接用于实验对比表）
    if results:
        n = len(results)  # 实际完成的评测回合数
        grasp_rate = sum(r["grasped"] for r in results) / n * 100.0  # 抓取成功率
        cross_rate = sum(r["crossed_barrier"] for r in results) / n * 100.0  # 越障成功率
        place_rate = sum(r["placed"] for r in results) / n * 100.0  # 放置入框率
        succ_rate = sum(r["success"] for r in results) / n * 100.0  # 端到端任务总成功率
        avg_ms = float(np.mean(infer_times_ms)) if infer_times_ms else 0.0  # 平均单步推理耗时（ms）
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0  # 等效推理帧率（Hz）
        print("\n========== Evaluation Summary ==========")
        print(f"Checkpoint : {CKPT_PATH}")
        print(f"Episodes   : {n} (OOD distractor={OOD_DISTRACTOR}, OOD pos={OOD_POSITION})")
        print(f"Grasp      : {grasp_rate:.1f}%")
        print(f"Cross      : {cross_rate:.1f}%")
        print(f"Place      : {place_rate:.1f}%")
        print(f"Success    : {succ_rate:.1f}% ({sum(r['success'] for r in results)}/{n})")
        print(f"Inference  : {avg_ms:.2f} ms/step ({fps:.1f} FPS)")
        print("========================================")


if __name__ == "__main__":
    main()
