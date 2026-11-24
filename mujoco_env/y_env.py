import random  # 导入 random 模块，用于在多任务模式下随机抽取自然语言指令
import glfw  # 导入 GLFW 库，用于读取 MuJoCo 窗口的键盘按键与游戏手柄（Gamepad）状态
import mujoco  # 导入 MuJoCo 原生库，用于无界面离屏渲染与底层接触碰撞检测
import numpy as np  # 导入 NumPy 数值计算库，用于三维坐标、旋转矩阵和关节角向量运算

from mujoco_env.ik import solve_ik  # 导入逆运动学（IK）求解器，将末端目标位姿解算为 6 轴关节角
from mujoco_env.mujoco_parser import MuJoCoParserClass  # 导入 MuJoCo 封装类，负责模型加载、物理步进、相机抓取与渲染
from mujoco_env.transforms import r2rpy, rpy2r  # 导入欧拉角（roll/pitch/yaw）与 3x3 旋转矩阵的相互转换函数
from mujoco_env.utils import (  # 导入通用几何与图像工具函数
    add_title_to_img,  # 在相机 RGB 图像顶部绘制标题文字并缩放尺寸
    rotation_matrix,  # 根据旋转轴和旋转角度生成 4x4 齐次旋转矩阵
    sample_xyzs,  # 在指定 XYZ 范围内随机采样满足最小间距约束的多个物体坐标
)


class SimpleEnv:
    """
    多物体语言条件跨障分拣仿真环境（兼容 ACT / SmolVLA / Pi0 训练与部署）。

    场景空间布局（以主视角相机 agentview 看过去为准）：
        - 左侧抓取区 (Y < 0)：随机生成苹果 (body_obj_apple) 与橙子 (body_obj_orange)；
                              当开启 ood_distractor=True 时额外在抓取区生成红牛罐 (body_obj_redbull)。
        - 中间矮隔板 (Y = 0)：高 6.5 cm 的半透明隔板，迫使机械臂执行“抬升 → 越障 → 放下”三维轨迹。
        - 右侧分拣区 (Y > 0)：蓝色分拣托盘 (tray_blue) 与绿色分拣托盘 (tray_green)。
    """

    # 预定义 4 条标准语言指令及其对应的目标物体、干扰物体和目标托盘（2 种水果 × 2 个颜色托盘）
    TASKS = [
        {  # 任务 0：把苹果放到蓝色托盘
            "instruction": "Pick up the apple and place it in the blue tray.",  # 标准英文任务描述（供 VLA 语言编码器输入）
            "target_obj": "body_obj_apple",  # 本任务应当抓取并放置的目标刚体名
            "other_obj": "body_obj_orange",  # 本任务中的非目标水果刚体名（用于检测是否误放）
            "target_tray": "tray_blue",  # 本任务对应的目标托盘刚体名
        },
        {  # 任务 1：把苹果放到绿色托盘
            "instruction": "Pick up the apple and place it in the green tray.",  # 标准英文任务描述
            "target_obj": "body_obj_apple",  # 目标物体：苹果
            "other_obj": "body_obj_orange",  # 非目标物体：橙子
            "target_tray": "tray_green",  # 目标托盘：绿色托盘
        },
        {  # 任务 2：把橙子放到蓝色托盘
            "instruction": "Pick up the orange and place it in the blue tray.",  # 标准英文任务描述
            "target_obj": "body_obj_orange",  # 目标物体：橙子
            "other_obj": "body_obj_apple",  # 非目标物体：苹果
            "target_tray": "tray_blue",  # 目标托盘：蓝色托盘
        },
        {  # 任务 3：把橙子放到绿色托盘
            "instruction": "Pick up the orange and place it in the green tray.",  # 标准英文任务描述
            "target_obj": "body_obj_orange",  # 目标物体：橙子
            "other_obj": "body_obj_apple",  # 非目标物体：苹果
            "target_tray": "tray_green",  # 目标托盘：绿色托盘
        },
    ]

    # 关闭 OOD 干扰物模式时，将红牛罐停放到所有相机视野之外的地面安全坐标 [x, y, z]
    DISTRACTOR_PARK_POS = np.array([-1.50, 0.0, 0.05], dtype=np.float64)

    # 动态物体刚体名到实际碰撞/视觉几何体名的映射（用于精确读取三维中心坐标）
    OBJ_GEOM_MAP = {
        "body_obj_apple": "geom_obj_apple",
        "body_obj_orange": "geom_obj_orange",
        "body_obj_redbull": "geom_obj_redbull",
    }

    def __init__(  # 环境初始化方法
        self,
        xml_path,  # MuJoCo 场景主 XML 文件路径
        action_type="eef_pose",  # 动作输入模式：'eef_pose'(遥操作末端增量) / 'joint_angle'(策略推理绝对关节角)
        seed=None,  # 随机种子：传入整数可固定物体初位，None 则每次重置随机化
        task_idx=None,  # 默认任务索引（0~3）；为 None 时每次 reset 随机抽取任务
        ood_distractor=False,  # 是否默认开启 OOD 干扰物（红牛罐）
        ood_position=False,  # 是否默认开启 OOD 扩展位置采样范围
        headless=False,  # 是否以无界面模式运行（True 时不弹 GLFW 窗口，适合批测与脚本验证）
        verbose=True,  # 是否打印底层模型信息与每次 reset 日志
    ):
        # 加载 MuJoCo XML 模型并创建底层仿真解析器实例
        self.env = MuJoCoParserClass(
            name="LanguageObstacleSorting", rel_xml_path=xml_path, verbose=verbose
        )
        self.action_type = action_type  # 保存动作控制模式
        self.default_task_idx = task_idx  # 保存默认任务索引
        self.ood_distractor = ood_distractor  # 保存默认 OOD 干扰物开关
        self.ood_position = ood_position  # 保存默认 OOD 位置扩展开关
        self.headless = headless  # 保存无界面模式标志
        self.verbose = verbose  # 保存日志开关标志
        self._offscreen_renderer = None  # 无界面模式下使用的 MuJoCo 离屏渲染器句柄

        # 缓存隔板几何体与静态支撑体 ID，供高频碰撞检测使用
        self._barrier_geom_id = self.env.model.geom("barrier_wall").id
        self._static_body_ids = {
            0,
            self.env.model.body("front_object_table").id,
            self.env.model.body("sorting_barrier").id,
        }

        self.joint_names = [  # OpenManipulator-Y 机械臂 6 个旋转关节的名称列表（与 omy.xml 保持一致）
            "joint1",  # 底座回转关节（绕 Z 轴）
            "joint2",  # 肩部俯仰关节（绕 Y 轴）
            "joint3",  # 肘部俯仰关节（绕 Y 轴）
            "joint4",  # 腕部俯仰关节（绕 Y 轴）
            "joint5",  # 腕部旋转关节（绕 Z 轴）
            "joint6",  # 末端法兰俯仰/摆动关节（绕 Y 轴）
        ]
        self.instruction = self.TASKS[0]["instruction"]  # 初始化当前语言指令文本为任务 0
        self.obj_target = self.TASKS[0]["target_obj"]  # 初始化当前目标物体刚体名
        self.obj_other = self.TASKS[0]["other_obj"]  # 初始化当前非目标物体刚体名
        self.tray_target = self.TASKS[0]["target_tray"]  # 初始化当前目标托盘刚体名

        self._gp_status_text = "Not Connected"  # 窗口叠加显示用的手柄连接状态文案

        if not self.headless:  # 若处于可视化交互模式
            self.init_viewer()  # 初始化 MuJoCo 交互渲染窗口与观察视角
        else:  # 若处于无界面模式（自动化数据生成 / 离线批测）
            self.env.reset()  # 仅重置底层仿真数据结构，不创建图形窗口
            # 创建与 XML 默认 offwidth/offheight (640x480) 一致的离屏渲染器
            self._offscreen_renderer = mujoco.Renderer(
                self.env.model, height=480, width=640
            )
            # 关闭天空盒以与可视化窗口 black_sky=True 保持视觉一致
            self._offscreen_renderer.scene.flags[
                mujoco.mjtRndFlag.mjRND_SKYBOX
            ] = False

        self.reset(  # 执行首次环境重置：归位机械臂、摆放物体并设定初始语言指令
            seed=seed,  # 传入初始随机种子
            task_idx=task_idx,  # 传入初始任务索引
            ood_distractor=ood_distractor,  # 传入 OOD 干扰物开关
            ood_position=ood_position,  # 传入 OOD 位置外推开关
        )

    def init_viewer(self):
        """初始化 MuJoCo 交互式可视化窗口参数"""
        self.env.reset()  # 先重置底层仿真时钟与状态
        self.env.init_viewer(  # 启动 GLFW 渲染窗口并配置观察相机
            distance=2.0,  # 主交互观察相机距离焦点的距离（单位：米）
            elevation=-30,  # 观察相机俯仰角（-30 度斜向下俯视桌面）
            transparent=False,  # 关闭半透明骨骼渲染
            black_sky=True,  # 使用纯黑背景，减少视觉干扰
            use_rgb_overlay=False,  # 不使用默认内置小窗，改由 render() 手动精确排版三路视角
            loc_rgb_overlay="top right",  # 默认 RGB 叠加小窗锚点位置
        )

    def set_instruction(self, given=None, task_idx=None):
        """
        设置当前 episode 的自然语言指令，并自动绑定目标物体与目标托盘。
        支持 3 种调用方式：
          1) 传入 task_idx (0~3)：精确选用标准任务列表中的某一项；
          2) 传入 given 文本：支持 OOD 同义改写指令（如 'Put the red fruit in the blue box'）；
          3) 均不传：若设置了 default_task_idx 则用默认任务，否则从 4 条标准任务中随机抽取。
        """
        if task_idx is not None:  # 优先级 1：按指定任务索引设置
            spec = self.TASKS[task_idx % len(self.TASKS)]  # 取模防止索引越界，获取任务配置字典
            self.instruction = spec["instruction"]  # 更新自然语言指令字符串
            self.obj_target = spec["target_obj"]  # 更新目标物体名
            self.obj_other = spec["other_obj"]  # 更新非目标物体名
            self.tray_target = spec["target_tray"]  # 更新目标托盘名
            return  # 设置完毕直接返回

        if given is not None:  # 优先级 2：解析自定义自然语言指令（用于语言泛化 OOD 测试）
            self.instruction = given  # 原样保存传入的文本指令，供 VLA 模型读取
            lower = given.lower()  # 转为小写以便大小写无关的关键字匹配
            if "apple" in lower or "red" in lower:  # 若指令提到苹果或红色水果
                self.obj_target = "body_obj_apple"  # 绑定目标物体为苹果
                self.obj_other = "body_obj_orange"  # 绑定非目标物体为橙子
            elif "orange" in lower or "citrus" in lower:  # 若指令提到橙子或柑橘类水果
                self.obj_target = "body_obj_orange"  # 绑定目标物体为橙子
                self.obj_other = "body_obj_apple"  # 绑定非目标物体为苹果
            else:  # 若指令未能指明有效目标物体
                raise ValueError(  # 抛出异常提醒调用方检查指令文本
                    f"Instruction must specify target object (apple/orange): {given}"
                )

            if "blue" in lower or "far" in lower or "top" in lower:  # 若指令指向蓝色/远端/上方托盘
                self.tray_target = "tray_blue"  # 绑定目标托盘为蓝色托盘
            elif "green" in lower or "near" in lower or "bottom" in lower:  # 若指令指向绿色/近端/下方托盘
                self.tray_target = "tray_green"  # 绑定目标托盘为绿色托盘
            else:  # 若指令未能指明有效目标托盘
                raise ValueError(  # 抛出异常提醒调用方检查指令文本
                    f"Instruction must specify target tray (blue/green): {given}"
                )
            return  # 自定义指令解析完成并返回

        if self.default_task_idx is not None:  # 优先级 3：若初始化时固定了默认任务索引
            spec = self.TASKS[self.default_task_idx % len(self.TASKS)]  # 取对应的标准任务
        else:  # 否则从 4 条标准任务中均匀随机抽取一条
            spec = random.choice(self.TASKS)
        self.instruction = spec["instruction"]  # 更新当前指令文本
        self.obj_target = spec["target_obj"]  # 更新目标物体名
        self.obj_other = spec["other_obj"]  # 更新非目标物体名
        self.tray_target = spec["target_tray"]  # 更新目标托盘名

    def _zero_freejoint_vel(self, body_name):
        """将指定自由物体（freejoint）的 6 维速度（线速度 + 角速度）清零，防止 reset 后残留动量"""
        jntadr = self.env.model.body(body_name).jntadr[0]  # 查询该刚体挂载的第一个关节索引
        if jntadr >= 0:  # 若该刚体确实绑定了有效关节
            dofadr = self.env.model.jnt_dofadr[jntadr]  # 获取该关节在全局速度数组 qvel 中的起始下标
            self.env.data.qvel[dofadr : dofadr + 6] = 0.0  # 将 3 维线速度与 3 维角速度全部置零

    def reset(  # 重置仿真环境到新一条 episode 的初始状态
        self,
        seed=None,  # 本次重置使用的随机种子（None 表示不重置随机状态）
        task_idx=None,  # 指定本轮任务索引（0~3），为 None 时按默认/随机策略选择
        instruction=None,  # 可选：直接指定本轮自定义自然语言指令（优先级高于随机）
        ood_distractor=None,  # 可选：覆盖是否启用红牛罐干扰物（None 时沿用实例默认值）
        ood_position=None,  # 可选：覆盖是否扩大抓取区采样范围（None 时沿用实例默认值）
    ):
        if seed is not None:  # 若传入了固定种子，则同时锁定 NumPy 和 Python 内置随机数生成器
            np.random.seed(seed)  # 锁定 NumPy 随机种子（控制物体坐标采样）
            random.seed(seed)  # 锁定 random 随机种子（控制任务随机抽取）

        if ood_distractor is None:  # 若未显式传入干扰物开关
            ood_distractor = self.ood_distractor  # 回退使用初始化时的默认开关
        if ood_position is None:  # 若未显式传入位置扩展开关
            ood_position = self.ood_position  # 回退使用初始化时的默认开关

        # 1. 使用 IK 将机械臂复位到桌边实验区中央上方（x=0.29, y=-0.02, z=1.02），夹爪垂直朝下
        # 采用明确的肘朝上（Elbow-Up）初值猜测，避免落入大臂平压桌面的肘朝下分支
        q_init = np.deg2rad([0, 10, 75, 5, 90, 0])  # IK 求解的肘朝上初始关节角猜测
        q_zero, _, _ = solve_ik(  # 调用逆运动学求解目标初位对应的 6 轴关节角
            env=self.env,  # MuJoCo 环境对象
            joint_names_for_ik=self.joint_names,  # 参与求解的 6 个臂关节名
            body_name_trgt="tcp_link",  # 以末端夹爪中心 tcp_link 为目标连杆
            q_init=q_init,  # 初始迭代猜测角度
            p_trgt=np.array([0.29, -0.02, 0.95]),  # 初始位置在桌边隔板上方安全高度，便于左右观察与移动
            R_trgt=rpy2r(np.deg2rad([90, 0.0, 90])),  # 末端垂直朝向桌面抓取的基准姿态旋转矩阵
        )
        # 通过前向运动学直接将机械臂摆放到初始关节角（不推进仿真计时器）
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)

        # 2. 在左侧桌边抓取区随机采样苹果、橙子（以及可选的 OOD 红牛罐干扰物）坐标
        if ood_position:  # OOD 位置外推测试模式：向四周扩展采样边界
            x_range = [0.17, 0.42]  # 扩展后的前后 X 范围（米）
            y_range = [-0.26, -0.08]  # 扩展后的左右 Y 范围（仍位于隔板左侧）
        else:  # 标准训练与标准测试模式
            x_range = [0.21, 0.39]  # 标准前后 X 范围（紧贴机械臂舒适工作半径）
            y_range = [-0.23, -0.10]  # 标准左右 Y 范围（远离隔板且居中于主相机左侧视野）

        n_active = 3 if ood_distractor else 2  # 开启干扰物时采 3 个不重叠点，否则仅采 2 个点
        obj_xyzs = sample_xyzs(  # 批量采样满足最小水平间距的物体初始三维坐标
            n_active,  # 需要采样的物体个数
            x_range=x_range,  # X 轴采样区间
            y_range=y_range,  # Y 轴采样区间
            z_range=[0.805, 0.805],  # 略高于桌面（z=0.800）5 mm 释放，避免初始穿模反弹
            min_dist=0.095,  # 任意两物体中心最小水平间距 9.5 cm，防止相互碰撞或夹取干涉
            xy_margin=0.0,  # 不额外内缩边界
        )

        # 摆放苹果并复位姿态与速度
        self.env.set_p_base_body(body_name="body_obj_apple", p=obj_xyzs[0, :])  # 设置苹果位置
        self.env.set_R_base_body(body_name="body_obj_apple", R=np.eye(3))  # 重置苹果朝向为单位阵（直立）
        self._zero_freejoint_vel("body_obj_apple")  # 清零苹果线/角速度

        # 摆放橙子并复位姿态与速度
        self.env.set_p_base_body(body_name="body_obj_orange", p=obj_xyzs[1, :])  # 设置橙子位置
        self.env.set_R_base_body(body_name="body_obj_orange", R=np.eye(3))  # 重置橙子朝向为直立
        self._zero_freejoint_vel("body_obj_orange")  # 清零橙子线/角速度

        # 根据 OOD 开关决定红牛罐放桌面抓取区还是停放到视野外
        if ood_distractor:  # 若启用干扰物
            self.env.set_p_base_body(body_name="body_obj_redbull", p=obj_xyzs[2, :])  # 放到第 3 个采样点
        else:  # 若未启用干扰物
            self.env.set_p_base_body(  # 移到相机看不见的场外停放区
                body_name="body_obj_redbull", p=self.DISTRACTOR_PARK_POS
            )
        self.env.set_R_base_body(body_name="body_obj_redbull", R=np.eye(3))  # 重置红牛罐直立
        self._zero_freejoint_vel("body_obj_redbull")  # 清零红牛罐速度

        self.env.forward(increase_tick=False)  # 刷新前向运动学使所有刚体位置立即生效

        # 3. 初始化机械臂控制目标缓存（6 轴臂关节 + 4 轴夹爪关节初始全张开为 0）
        self.q = np.concatenate([q_zero, np.array([0.0] * 4)])  # 拼接 10 维执行器目标向量
        self.p0, self.R0 = self.env.get_pR_body(body_name="tcp_link")  # 缓存末端初始位置 p0 与姿态 R0，作为增量遥操作基准

        # 4. 空跑 80 步物理仿真（约 0.16 秒），让水果平稳落座在桌面上完全静止
        for _ in range(80):  # 循环推进物理仿真
            self.step_env()  # 施加初始目标控制量并步进一步

        # 落稳后记录 9 维物体初始位姿向量（apple_xyz + orange_xyz + redbull_xyz），写入数据集用于精准回放
        p_apple, p_orange, p_redbull = self.get_obj_pose()  # 读取三个物体稳定后的三维世界坐标
        self.obj_init_pose = np.concatenate(  # 拼接为连续 9 维 float32 数组
            [p_apple, p_orange, p_redbull], dtype=np.float32
        )

        # 5. 设置本轮语言指令并重置夹爪逻辑状态
        self.set_instruction(given=instruction, task_idx=task_idx)  # 绑定本轮指令与目标物体/托盘
        self.gripper_state = False  # 夹爪初始逻辑状态设为张开（False=0.0, True=1.0）
        if self.verbose:
            print(f"DONE INITIALIZATION | Task: {self.instruction}")  # 在终端打印本轮任务指令方便操作者查看

    def step(self, action):
        """
        将上层动作指令转换为底层 10 维执行器目标 self.q（无返回值，状态请用 get_joint_state() 读取）。
        args:
            action: 长度为 7 的 NumPy 数组
                - eef_pose 模式: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                - joint_angle 模式: [j1, j2, j3, j4, j5, j6, gripper]
        """
        if self.action_type == "eef_pose":  # 末端位姿增量模式（用于人工键盘/手柄遥操作）
            q = self.env.get_qpos_joints(joint_names=self.joint_names)  # 读取当前实际 6 轴关节角作为 IK 迭代初值
            self.p0 += action[:3]  # 将位置增量 [dx, dy, dz] 累加到目标末端位置
            # 对末端目标位置做工作空间安全限幅，防止持续按下 RT/摇杆导致 p0 钻入桌面以下引发 IK 奇异翻折
            self.p0[0] = np.clip(self.p0[0], 0.14, 0.46)  # X 轴安全工作范围（近桌边到绿盘前缘）
            self.p0[1] = np.clip(self.p0[1], -0.29, 0.29)  # Y 轴安全工作范围（左抓取区到右分拣区）
            self.p0[2] = np.clip(self.p0[2], 0.822, 1.15)  # Z 轴安全范围（最低贴近桌面抓取，不钻桌底）
            self.R0 = self.R0.dot(rpy2r(action[3:6]))  # 将姿态增量旋转矩阵右乘到当前目标旋转矩阵（工具系增量旋转）
            q, _, _ = solve_ik(  # 通过阻尼最小二乘 IK 求解满足新末端位姿的目标关节角
                env=self.env,  # MuJoCo 环境实例
                joint_names_for_ik=self.joint_names,  # 求解的 6 个关节
                body_name_trgt="tcp_link",  # 目标末端连杆名
                q_init=q,  # 当前关节角初值
                p_trgt=self.p0,  # 目标末端三维位置
                R_trgt=self.R0,  # 目标末端旋转矩阵
                max_ik_tick=50,  # 最大 IK 迭代步数
                ik_stepsize=1.0,  # IK 更新步长系数
                ik_eps=1e-2,  # 误差收敛判定阈值
                ik_th=np.radians(5.0),  # 单步迭代关节角最大变化幅度限制（5 度），防止奇异点跳变
                render=False,  # 求解过程不触发渲染
                verbose_warning=False,  # 关闭迭代未完全收敛时的刷屏警告
            )
        elif self.action_type == "joint_angle":  # 绝对关节角模式（用于 ACT / SmolVLA / Pi0 策略推理及数据回放）
            q = action[:-1]  # 直接取动作前 6 维作为目标关节角
        else:  # 传入了不支持的动作模式名
            raise ValueError("action_type not recognized")  # 抛出值异常

        # 夹爪控制：将第 7 维标量指令映射到 4 个夹爪驱动关节 [rh_r1, rh_r2, rh_l1, rh_l2]
        # r2/l2 联动系数修正为 1.0（原代码 0.8 会导致指尖内倾 11°~19° 挤飞圆形水果），并按各自限位限幅
        g = float(action[-1])  # 提取标量夹爪开合指令（0.0=全开，1.0=闭合）
        gripper_cmd = np.array(  # 构造 4 维夹爪执行器目标向量
            [
                np.clip(g, 0.0, 1.1),  # 右近端关节 rh_r1（限位 [0, 1.1]）
                np.clip(g, 0.0, 1.0),  # 右远端关节 rh_r2（限位 [0, 1.0]，保持指尖平行）
                np.clip(g, 0.0, 1.1),  # 左近端关节 rh_l1（限位 [0, 1.1]）
                np.clip(g, 0.0, 1.0),  # 左远端关节 rh_l2（限位 [0, 1.0]，保持指尖平行）
            ],
            dtype=np.float64,
        )
        self.q = np.concatenate([q, gripper_cmd])  # 拼接 6 轴臂关节 + 4 轴夹爪指令为 10 维底层控制量

    def step_env(self):
        """将当前控制目标 self.q 写入 MuJoCo 执行器并推进一个物理仿真步长（dt=0.002s）"""
        self.env.step(self.q)  # 底层执行一次 mj_step

    def _render_offscreen_cam(self, cam_name: str):
        """使用 MuJoCo 离屏渲染器抓取指定固定相机画面，并保持黑背景设置一致"""
        self._offscreen_renderer.update_scene(self.env.data, camera=cam_name)
        self._offscreen_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        return self._offscreen_renderer.render().copy()

    def grab_image(self, include_side=None):
        """从场景中的固定相机渲染抓取 RGB 图像，返回主视角与腕部视角"""
        if include_side is None:
            include_side = not self.headless

        if self.headless:
            if self._offscreen_renderer is None:
                self._offscreen_renderer = mujoco.Renderer(
                    self.env.model, height=480, width=640
                )
            self.rgb_agent = self._render_offscreen_cam("agentview")
            self.rgb_ego = self._render_offscreen_cam("egocentric")
            self.rgb_side = (
                self._render_offscreen_cam("sideview") if include_side else None
            )
            return self.rgb_agent, self.rgb_ego

        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name="agentview")  # 抓取第三人称主视角图像
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name="egocentric")  # 抓取腕部第一人称视角图像
        self.rgb_side = (  # 抓取侧面辅助视角图像（供遥操作显示）
            self.env.get_fixed_cam_rgb(cam_name="sideview") if include_side else None
        )
        return self.rgb_agent, self.rgb_ego  # 返回策略训练/推理所需的两路视角

    def close(self):
        """统一释放离屏渲染器或交互式 GLFW 渲染窗口资源"""
        if getattr(self, "_offscreen_renderer", None) is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None
        elif getattr(self.env, "use_mujoco_viewer", False):
            self.env.close_viewer()

    def render(self, teleop=False, idx=None):
        """渲染 MuJoCo 可视化窗口，并绘制末端指示器、多路相机小窗以及任务/手柄文字提示"""
        self.env.plot_time()  # 在场景中显示当前仿真时间
        p_current, R_current = self.env.get_pR_body(body_name="tcp_link")  # 获取当前末端执行器的世界位置与旋转矩阵
        R_current = R_current @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])  # 调整胶囊体局部轴向以与夹爪朝向一致
        self.env.plot_sphere(p=p_current, r=0.015, rgba=[0.95, 0.05, 0.05, 0.45])  # 在末端中心绘制半透明红色标记球
        self.env.plot_capsule(  # 绘制指示柱（半长 0.35m，从高处即可直达桌面精确瞄准落点）
            p=p_current, R=R_current, r=0.006, h=0.35, rgba = [1.0, 1.0, 1.0, 0.5]   # 白色指示线
        )
        if idx is not None:  # 若传入了当前 episode 编号
            self.env.plot_T(  # 在场景上方悬浮绘制 Episode 编号文字
                p=np.array([0.1, 0.0, 1.05]),  # 文字锚点三维坐标
                label=f"Episode {idx}",  # 显示文本内容
                plot_axis=False,  # 不绘制坐标轴箭头
                plot_sphere=False,  # 不绘制原点小球
            )

        rgb_egocentric_view = add_title_to_img(  # 给腕部视角图像加上标题栏并缩放到 640x480
            self.rgb_ego, text="Egocentric View", shape=(640, 480)
        )
        rgb_agent_view = add_title_to_img(  # 给主视角图像加上标题栏并缩放到 640x480
            self.rgb_agent, text="Agent View", shape=(640, 480)
        )
        self.env.viewer_rgb_overlay(rgb_agent_view, loc="top right")  # 将主视角小窗叠加在窗口右上角
        self.env.viewer_rgb_overlay(rgb_egocentric_view, loc="bottom right")  # 将腕部视角小窗叠加在窗口右下角

        if teleop:  # 遥操作采集模式下额外叠加侧视角与设备状态文字
            rgb_side_view = add_title_to_img(  # 给侧视角图像加上标题栏
                self.rgb_side, text="Side View", shape=(640, 480)
            )
            self.env.viewer_rgb_overlay(rgb_side_view, loc="top left")  # 将侧视角小窗叠加在窗口左上角
            self.env.viewer_text_overlay(  # 叠加显示手柄连接状态与速度档位（NORMAL / SLOW）
                text1="Gamepad", text2=self._gp_status_text
            )
            self.env.viewer_text_overlay(  # 叠加显示夹爪当前逻辑开关状态（OPEN / CLOSED）
                text1="Gripper",
                text2="CLOSED" if self.gripper_state else "OPEN",
            )
            self.env.viewer_text_overlay(  # 叠加显示当前按下的键盘按键列表
                text1="Key Pressed",
                text2="%s" % (self.env.get_key_pressed_list()),
            )
        if getattr(self, "instruction", None) is not None:  # 只要存在任务语言指令就始终在画面中展示
            self.env.viewer_text_overlay(  # 叠加显示当前自然语言指令，便于人工采集和 VLA 部署观察
                text1="Task", text2=self.instruction
            )
        self.env.render()  # 提交 OpenGL 渲染命令并交换缓冲区刷新屏幕

    def get_joint_state(self):
        """读取机械臂当前真实关节角与二值化夹爪状态，返回 7 维向量 [j1..j6, gripper]"""
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)  # 读取 6 个臂关节当前实际角度（弧度）
        gripper = self.env.get_qpos_joint("rh_r1")  # 读取夹爪主驱动关节 rh_r1 当前角度
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0  # 角度 > 0.5 视为闭合（1.0），否则视为张开（0.0）
        return np.concatenate([qpos, [gripper_cmd]], dtype=np.float32)  # 拼接并返回 7 维 float32 状态向量

    def teleop_robot(self):
        """
        融合键盘与游戏手柄的遥操作输入读取函数（两者同时生效、增量叠加）。
        returns:
            action: np.ndarray(7,) -> [dx, dy, dz, droll, dpitch, dyaw, gripper]
            reset: bool -> True 表示用户触发了重置当前 episode
        """
        dpos = np.zeros(3)  # 初始化本帧三维平移增量 [dx, dy, dz]
        drot = np.eye(3)  # 初始化本帧三维旋转增量矩阵为单位阵
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):  # 长按键盘 S：沿 +X 移动（朝向主相机）
            dpos += np.array([0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):  # 长按键盘 W：沿 -X 移动（朝向机械臂底座）
            dpos += np.array([-0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):  # 长按键盘 A：沿 -Y 移动（主视角向左）
            dpos += np.array([0.0, -0.007, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):  # 长按键盘 D：沿 +Y 移动（主视角向右）
            dpos += np.array([0.0, 0.007, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):  # 长按键盘 R：沿 +Z 上升
            dpos += np.array([0.0, 0.0, 0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):  # 长按键盘 F：沿 -Z 下降
            dpos += np.array([0.0, 0.0, -0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):  # 左方向键：绕末端 Y 轴正向旋转
            drot = drot @ rotation_matrix(angle=0.03, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):  # 右方向键：绕末端 Y 轴负向旋转
            drot = drot @ rotation_matrix(angle=-0.03, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):  # 下方向键：绕末端 X 轴正向旋转
            drot = drot @ rotation_matrix(angle=0.03, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):  # 上方向键：绕末端 X 轴负向旋转
            drot = drot @ rotation_matrix(angle=-0.03, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):  # 长按键盘 Q：绕末端 Z 轴正向偏航旋转
            drot = drot @ rotation_matrix(angle=0.03, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):  # 长按键盘 E：绕末端 Z 轴负向偏航旋转
            drot = drot @ rotation_matrix(angle=-0.03, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):  # 单击键盘 Z：立即返回零动作与重置信号 True
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):  # 单击空格键：切换夹爪开/合状态
            self.gripper_state = not self.gripper_state

        # ---------- 手柄控制（与键盘输入叠加生效，无手柄时自动跳过） ----------
        gp = self._read_gamepad()  # 轮询读取当前帧手柄动作字典
        if gp is not None:  # 若检测到已连接的标准手柄
            dpos = dpos + gp["dpos"]  # 叠加手柄产生的三维位置增量
            if gp["rot_axis_angle"] is not None:  # 若手柄产生旋转增量矩阵
                drot = drot @ gp["rot_axis_angle"]  # 右乘复合到总旋转增量矩阵中
            if gp["reset"]:  # 若按下手柄 Y 或 Start 键请求重置
                return np.zeros(7, dtype=np.float32), True  # 返回零动作和重置信号
            if gp["toggle_gripper"]:  # 若触发手柄 A 键上升沿
                self.gripper_state = not self.gripper_state  # 切换夹爪开/合状态

        drot = r2rpy(drot)  # 将复合后的 3x3 旋转增量矩阵转换为欧拉角 [droll, dpitch, dyaw]
        action = np.concatenate(  # 拼接 7 维完整遥操作动作向量
            [dpos, drot, np.array([self.gripper_state], dtype=np.float32)],
            dtype=np.float32,
        )
        return action, False  # 返回动作向量及非重置标志 False

    def _read_gamepad(self, deadzone=0.15, pos_step=0.007, rot_step=0.03):
        """
        使用 GLFW 内置标准 Gamepad 映射读取手柄状态（支持 Xbox / PS 等主流手柄）。

        按键映射表：
            左摇杆      : XY 平面平移（上推=-X 同键盘 W，右推=+Y 同键盘 D）
            LT / RT     : Z 轴上升 / 下降（按压越深移动越快）
            右摇杆 X / Y: 绕 Z 轴偏航 / 绕 X 轴俯仰
            LB / RB     : 绕 Y 轴旋转（LB 同 ←，RB 同 →）
            A 键        : 夹爪开/合切换（边沿触发）
            B 键（按住）: 慢速精细微调模式（整体速度降至 0.4×）
            Y 键 / Start: 重置当前轨迹（边沿触发）
        """
        if not hasattr(self, "_gp_prev"):  # 首次调用时初始化手柄边沿检测与状态缓存
            self._gp_prev = {"A": 0, "Y": 0, "START": 0}  # 记录上一帧功能键电平，用于检测按下瞬间
            self._gp_warned = False  # 记录提示打印状态，防止每帧重复刷屏
            self._gp_trig_ready = [False, False]  # 分别标记 LT / RT 扳机轴是否已离开未初始化的 0.0 状态

        jid = None  # 初始化有效手柄端口 ID
        for j in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):  # 遍历 GLFW 支持的 16 个手柄槽位
            if glfw.joystick_present(j) and glfw.joystick_is_gamepad(j):  # 查找第一个在线且符合标准映射的手柄
                jid = j  # 锁定该手柄 ID
                break  # 找到后立即退出遍历
        if jid is None:  # 若当前未检测到可用手柄
            if not self._gp_warned:  # 仅在首次未检测到时打印一次终端提示
                print("[Gamepad] 未检测到手柄，仅使用键盘控制")
                self._gp_warned = True  # 标记已提示
            self._gp_status_text = "Not Connected (Keyboard Only)"  # 更新窗口叠加提示文字
            return None  # 返回 None，由调用方继续仅用键盘控制

        state = glfw.get_gamepad_state(jid)  # 读取该手柄当前所有按键与摇杆轴的瞬时快照
        if state is None:  # 若读取失败（如刚拔掉手柄）
            return None  # 安全返回 None

        gp_name = glfw.get_gamepad_name(jid)  # 获取手柄设备名称字符串
        if self._gp_warned != "connected":  # 若从断连/未连接状态首次切换为已连接
            print(f"[Gamepad] 已连接: {gp_name}")  # 打印连接成功提示与设备名
            self._gp_warned = "connected"  # 标记为已连接状态

        b, ax = state.buttons, state.axes  # 解包 15 维按键状态列表与 6 维模拟轴数值列表

        def dz(v):  # 死区滤波并线性重映射函数：消除摇杆零点微小漂移
            if abs(v) < deadzone:  # 落入死区范围内直接归零（避免误触发开始录制）
                return 0.0
            # 将 [deadzone, 1.0] 平滑重映射到 [0.0, 1.0]，保持符号不变
            return float(np.sign(v) * (abs(v) - deadzone) / (1.0 - deadzone))

        def read_trigger(axis_val, idx):  # 扳机轴归一化读取（防未触碰前输出 0.0 引起 Z 轴漂移）
            if not self._gp_trig_ready[idx]:  # 若该扳机尚未完成首次校准
                if axis_val != 0.0:  # 一旦收到非 0 值（如 XInput 默认静止值 -1.0 或被按动）
                    self._gp_trig_ready[idx] = True  # 标记该扳机已就绪
                else:  # 若在首次触碰前驱动恒报 0.0
                    return 0.0  # 强制按未按下（0.0）处理
            return dz((axis_val + 1.0) / 2.0)  # 将 GLFW 扳机范围 [-1.0, +1.0] 归一化到 [0.0, 1.0] 并过死区

        slow_mode = bool(b[glfw.GAMEPAD_BUTTON_B])  # 检测是否按住 B 键启用慢速精细档
        speed_scale = 0.4 if slow_mode else 1.0  # 慢速档将步长缩放为 40%，便于精准对准水果
        self._gp_status_text = f"{gp_name} ({'SLOW' if slow_mode else 'NORMAL'})"  # 更新画面手柄状态文字

        lx = dz(ax[glfw.GAMEPAD_AXIS_LEFT_X])  # 读取左摇杆水平轴（-1=左，+1=右，对应 Y 轴）
        ly = dz(ax[glfw.GAMEPAD_AXIS_LEFT_Y])  # 读取左摇杆垂直轴（-1=上推，+1=下推，对应 X 轴）
        rx = dz(ax[glfw.GAMEPAD_AXIS_RIGHT_X])  # 读取右摇杆水平轴（控制绕 Z 轴偏航）
        ry = dz(ax[glfw.GAMEPAD_AXIS_RIGHT_Y])  # 读取右摇杆垂直轴（控制绕 X 轴俯仰）
        lt = read_trigger(ax[glfw.GAMEPAD_AXIS_LEFT_TRIGGER], 0)  # 读取左扳机 LT（控制上升 +Z）
        rt = read_trigger(ax[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER], 1)  # 读取右扳机 RT（控制下降 -Z）
        dz_up = lt - rt  # 合成 Z 轴归一化速度指令：按 LT 为正（上升），按 RT 为负（下降）

        p_step = pos_step * speed_scale  # 计算本帧实际平移最大步长（米/帧）
        r_step = rot_step * speed_scale  # 计算本帧实际旋转最大步长（弧度/帧）
        dpos = np.array([ly * p_step, lx * p_step, dz_up * p_step])  # 组装本帧位置增量 [dx, dy, dz]

        R = np.eye(3)  # 初始化本帧手柄旋转增量矩阵为单位阵
        if rx != 0.0:  # 右摇杆左右推动：绕 Z 轴旋转
            R = R @ rotation_matrix(angle=-rx * r_step, direction=[0.0, 0.0, 1.0])[:3, :3]
        if ry != 0.0:  # 右摇杆上下推动：绕 X 轴旋转
            R = R @ rotation_matrix(angle=ry * r_step, direction=[1.0, 0.0, 0.0])[:3, :3]
        # 左/右肩键（LB / RB）差值：按 LB 为 +1（同键盘 ←），按 RB 为 -1（同键盘 →）
        bumper = float(b[glfw.GAMEPAD_BUTTON_LEFT_BUMPER]) - float(
            b[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER]
        )
        if bumper != 0.0:  # 肩键被按下时绕 Y 轴旋转
            R = R @ rotation_matrix(angle=bumper * r_step, direction=[0.0, 1.0, 0.0])[:3, :3]

        cur = {  # 采集当前帧功能键状态
            "A": b[glfw.GAMEPAD_BUTTON_A],  # A 键：切换夹爪
            "Y": b[glfw.GAMEPAD_BUTTON_Y],  # Y 键：重置轨迹
            "START": b[glfw.GAMEPAD_BUTTON_START],  # Start 键：重置轨迹
        }
        # 上升沿检测：仅在“本帧按下且上一帧未按下”时触发一次 True，防止按住时反复翻转
        pressed = {k: bool(cur[k] and not self._gp_prev[k]) for k in cur}
        self._gp_prev = cur  # 更新上一帧按键记录供下一帧比对
        return {  # 返回封装好的手柄动作解析结果
            "dpos": dpos,  # 三维平移增量
            "rot_axis_angle": R,  # 3x3 旋转增量矩阵
            "toggle_gripper": pressed["A"],  # 本帧是否触发夹爪切换
            "reset": pressed["Y"] or pressed["START"],  # 本帧是否触发轨迹重置
        }

    def _is_in_tray(self, obj_pos, tray_name):
        """判断给定物体坐标是否位于指定分拣托盘的内部有效空间且贴近托盘底面"""
        p_tray = self.env.get_p_body(tray_name)  # 获取目标托盘中心的世界坐标
        in_xy = (  # 检查物体中心在 X、Y 方向上是否都落入托盘 12cm×12cm 有效区域内（±6.2cm）
            abs(obj_pos[0] - p_tray[0]) < 0.062
            and abs(obj_pos[1] - p_tray[1]) < 0.062
        )
        # 检查物体底部高度是否落在托盘底面附近（底面 z=0.804，允许 ±数厘米容差，排除仍在半空的情况）
        in_z = 0.795 <= obj_pos[2] <= 0.840
        return in_xy and in_z  # 水平与高度同时满足才判定为在托盘内

    def _get_body_speed(self, body_name):
        """读取指定自由物体当前的线速度模长（m/s）与角速度模长（rad/s）"""
        jntadr = self.env.model.body(body_name).jntadr[0]  # 获取该物体自由关节索引
        dofadr = self.env.model.jnt_dofadr[jntadr]  # 获取其在全局速度向量 qvel 中的起始偏移
        vel = self.env.data.qvel[dofadr : dofadr + 6]  # 切片取出 6 维广义速度（前 3 线速度 + 后 3 角速度）
        return float(np.linalg.norm(vel[:3])), float(np.linalg.norm(vel[3:]))  # 返回线速度与角速度大小

    def check_success(self):
        """
        严格的自动任务成功判定（全部 5 项条件同时满足才返回 True）：
          1. 目标物体落在目标托盘内部且高度贴近盘底；
          2. 非目标水果未被误放入同一目标托盘；
          3. 目标物体已基本静止（线速度 < 0.05 m/s）；
          4. 夹爪已完全张开释放物体（rh_r1 < 0.12）；
          5. 机械臂末端已撤离抬升至安全高度（z > 0.90 m）。
        """
        p_target = self.env.get_p_body(self.obj_target)  # 读取目标物体坐标
        p_other = self.env.get_p_body(self.obj_other)  # 读取另一非目标水果坐标

        if not self._is_in_tray(p_target, self.tray_target):  # 条件 1：目标必须在目标托盘内
            return False
        if self._is_in_tray(p_other, self.tray_target):  # 条件 2：防止把两个水果一起扫进托盘作弊
            return False

        lin_speed, _ = self._get_body_speed(self.obj_target)  # 读取目标物体当前线速度
        if lin_speed >= 0.05:  # 条件 3：必须落稳静止，不能处于弹跳或滑动中
            return False

        if float(self.env.get_qpos_joint("rh_r1")[0]) >= 0.12:  # 条件 4：夹爪必须已松开
            return False

        p_tcp_z = float(self.env.get_p_body("tcp_link")[2])  # 读取夹爪末端当前高度 z
        if p_tcp_z <= 0.90:  # 条件 5：末端必须抬离托盘至 0.90m 以上
            return False

        return True  # 所有条件满足，判定本次演示/评测成功

    def get_obj_geom_center(self, body_name: str) -> np.ndarray:
        """读取物体实际几何体（geom）的世界三维中心坐标，避免网格偏移与底部原点误差"""
        geom_name = self.OBJ_GEOM_MAP.get(body_name)
        if geom_name is not None:
            return self.env.data.geom(geom_name).xpos.copy()
        return self.env.get_p_body(body_name).copy()

    def check_barrier_collision(self) -> bool:
        """检测机械臂或任意动态物体是否与中间隔板（barrier_wall）发生物理接触"""
        for i in range(self.env.data.ncon):
            c = self.env.data.contact[i]
            if c.geom1 == self._barrier_geom_id:
                other_geom = c.geom2
            elif c.geom2 == self._barrier_geom_id:
                other_geom = c.geom1
            else:
                continue
            other_body = int(self.env.model.geom_bodyid[other_geom])
            if other_body not in self._static_body_ids:
                return True
        return False

    def get_eval_diagnostics(self) -> dict:
        """读取细粒度分阶段评测状态，用于抓取/跨障/放盘阶段统计与失败归因分析"""
        other_tray = "tray_green" if self.tray_target == "tray_blue" else "tray_blue"
        p_target_body = self.env.get_p_body(self.obj_target)
        p_other_body = self.env.get_p_body(self.obj_other)
        p_target_geom = self.get_obj_geom_center(self.obj_target)
        p_redbull_body = self.env.get_p_body("body_obj_redbull")
        p_tcp = self.env.get_p_body("tcp_link")
        redbull_active = bool(p_redbull_body[0] > -0.5)
        return {
            "tcp_z": float(p_tcp[2]),
            "tcp_to_target_xy": float(np.linalg.norm(p_tcp[:2] - p_target_geom[:2])),
            "target_z": float(p_target_body[2]),
            "target_y": float(p_target_body[1]),
            "other_z": float(p_other_body[2]),
            "redbull_active": redbull_active,
            "redbull_z": float(p_redbull_body[2]) if redbull_active else 0.0,
            "target_in_target_tray": bool(self._is_in_tray(p_target_body, self.tray_target)),
            "target_in_wrong_tray": bool(self._is_in_tray(p_target_body, other_tray)),
            "other_in_target_tray": bool(self._is_in_tray(p_other_body, self.tray_target)),
            "other_in_wrong_tray": bool(self._is_in_tray(p_other_body, other_tray)),
            "success": bool(self.check_success()),
        }

    def get_obj_pose(self):
        """读取并返回苹果、橙子、红牛罐当前的三维世界坐标"""
        p_apple = self.env.get_p_body("body_obj_apple")  # 读取苹果坐标 [x, y, z]
        p_orange = self.env.get_p_body("body_obj_orange")  # 读取橙子坐标 [x, y, z]
        p_redbull = self.env.get_p_body("body_obj_redbull")  # 读取红牛罐坐标 [x, y, z]
        return p_apple, p_orange, p_redbull  # 按固定顺序返回三元组

    def set_obj_pose(self, p_apple, p_orange, p_redbull=None):
        """将苹果、橙子与可选红牛罐直接复位到指定坐标（供数据集回放和固定初始态评测使用）"""
        self.env.set_p_base_body(body_name="body_obj_apple", p=p_apple)  # 设置苹果三维坐标
        self.env.set_R_base_body(body_name="body_obj_apple", R=np.eye(3))  # 复位苹果为竖直朝向
        self._zero_freejoint_vel("body_obj_apple")  # 清零苹果速度

        self.env.set_p_base_body(body_name="body_obj_orange", p=p_orange)  # 设置橙子三维坐标
        self.env.set_R_base_body(body_name="body_obj_orange", R=np.eye(3))  # 复位橙子为竖直朝向
        self._zero_freejoint_vel("body_obj_orange")  # 清零橙子速度

        if p_redbull is None:  # 若未提供红牛罐坐标
            p_redbull = self.DISTRACTOR_PARK_POS  # 默认停放到视野外安全区
        self.env.set_p_base_body(body_name="body_obj_redbull", p=p_redbull)  # 设置红牛罐三维坐标
        self.env.set_R_base_body(body_name="body_obj_redbull", R=np.eye(3))  # 复位红牛罐为竖直朝向
        self._zero_freejoint_vel("body_obj_redbull")  # 清零红牛罐速度

        self.env.forward(increase_tick=False)  # 立即更新前向运动学几何位置
        self.step_env()  # 推进一帧仿真刷新接触状态

