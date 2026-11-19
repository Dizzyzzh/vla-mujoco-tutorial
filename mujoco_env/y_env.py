import numpy as np  # 数值计算与矩阵运算库
import xml.etree.ElementTree as ET  # XML 解析库（预留用于场景 XML 处理）
from mujoco_env.mujoco_parser import MuJoCoParserClass  # 封装 MuJoCo 模型加载、仿真步进、相机与渲染的解析器类
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img  # 采样、旋转矩阵与图像标题绘制等工具函数
from mujoco_env.ik import solve_ik  # 逆运动学（IK）求解器
from mujoco_env.transforms import rpy2r, r2rpy  # 欧拉角（roll, pitch, yaw）与旋转矩阵之间的相互转换
import copy  # 用于深拷贝关节角度数组
import glfw  # GLFW 键盘事件常量定义（用于遥操作按键检测）


class SimpleEnv:
    def __init__(
        self, xml_path, action_type="eef_pose", state_type="joint_angle", seed=None
    ):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name="Tabletop", rel_xml_path=xml_path)  # 加载 MuJoCo 桌面场景 XML 模型
        self.action_type = action_type  # 记录动作空间类型（末端增量/关节增量/绝对关节角）
        self.state_type = state_type  # 记录返回的状态空间类型（关节角/末端位姿/关节增量）

        self.joint_names = [  # 机械臂 6 个旋转关节的名称列表
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        self.init_viewer()  # 初始化 MuJoCo 可视化窗口
        self.reset(seed)  # 重置机械臂与桌面物体到初始状态

    def init_viewer(self):
        """
        Initialize the viewer
        """
        self.env.reset()  # 重置 MuJoCo 内部状态
        self.env.init_viewer(
            distance=2.0,  # 观察相机距离目标点的距离
            elevation=-30,  # 相机俯仰角（-30 度俯视）
            transparent=False,  # 关闭半透明渲染
            black_sky=True,  # 使用纯黑背景天空
            use_rgb_overlay=False,  # 初始化时不启用自带的默认 RGB 叠加层（由 render 手动控制）
            loc_rgb_overlay="top right",  # 默认 RGB 叠加层位置
        )

    def reset(self, seed=None):
        """
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        """
        if seed is not None:  # 若指定了随机种子，则固定随机数生成器
            np.random.seed(seed)
        q_init = np.deg2rad([0, 0, 0, 0, 0, 0])  # IK 求解的初始关节角猜测（全零位姿）
        q_zero, ik_err_stack, ik_info = solve_ik(
            env=self.env,  # MuJoCo 环境实例
            joint_names_for_ik=self.joint_names,  # 参与 IK 求解的 6 个关节名称
            body_name_trgt="tcp_link",  # 目标末端执行器连杆名称
            q_init=q_init,  # ik from zero pose
            p_trgt=np.array([0.3, 0.0, 1.0]),  # 末端初始目标位置 [x, y, z]
            R_trgt=rpy2r(np.deg2rad([90, -0.0, 90])),  # 末端初始目标姿态旋转矩阵
        )
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)  # 通过前向运动学将机械臂直接摆到目标初位

        # Set object positions
        obj_names = self.env.get_body_names(prefix="body_obj_")  # 获取场景中所有以 "body_obj_" 开头的物体刚体名
        n_obj = len(obj_names)  # 物体总数量（本场景中为杯子和盘子）
        obj_xyzs = sample_xyzs(
            n_obj,  # 采样物体数量
            x_range=[+0.24, +0.4],  # x 方向采样范围
            y_range=[-0.2, +0.2],  # y 方向采样范围
            z_range=[0.82, 0.82],  # 桌面高度 z 坐标
            min_dist=0.2,  # 物体之间的最小水平间距，防止碰撞重叠
            xy_margin=0.0,  # 边界余量
        )
        for obj_idx in range(n_obj):  # 遍历每个物体设置其初始位姿
            self.env.set_p_base_body(
                body_name=obj_names[obj_idx], p=obj_xyzs[obj_idx, :]  # 设置物体基准位置
            )
            self.env.set_R_base_body(body_name=obj_names[obj_idx], R=np.eye(3, 3))  # 设置物体初始朝向为单位矩阵
        self.env.forward(increase_tick=False)  # 刷新前向运动学状态

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)  # 记录上一时刻 6 轴关节角（用于 delta_joint_angle 模式）
        self.q = np.concatenate([q_zero, np.array([0.0] * 4)])  # 拼接 6 个臂关节角 + 4 个夹爪关节角（初始张开为 0）
        self.p0, self.R0 = self.env.get_pR_body(body_name="tcp_link")  # 记录末端执行器初始位置与旋转矩阵，供增量控制参考
        mug_init_pose, plate_init_pose = self.get_obj_pose()  # 获取杯子与盘子的初始三维坐标
        self.obj_init_pose = np.concatenate(
            [mug_init_pose, plate_init_pose], dtype=np.float32  # 拼接为 6 维物体初始位姿向量保存
        )
        for _ in range(100):  # 空跑 100 步物理仿真让物体稳定落在桌面上
            self.step_env()
        print("DONE INITIALIZATION")  # 打印初始化完成提示
        self.gripper_state = False  # 夹爪状态标志位（False=张开，True=闭合）
        self.past_chars = []  # 预留按键历史记录列表

    def step(self, action):
        """
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
        returns:
            state: np.array, state of the environment after taking the action
                - ee_pose: [px,py,pz,r,p,y]
                - joint_angle: [j1,j2,j3,j4,j5,j6]

        """
        if self.action_type == "eef_pose":  # 末端位姿增量控制模式（用于键盘遥操作）
            q = self.env.get_qpos_joints(joint_names=self.joint_names)  # 读取当前 6 轴关节角作为 IK 初值
            self.p0 += action[:3]  # 累加位置增量 [dx, dy, dz] 到目标末端位置
            self.R0 = self.R0.dot(rpy2r(action[3:6]))  # 右乘旋转增量矩阵更新目标末端姿态
            q, ik_err_stack, ik_info = solve_ik(
                env=self.env,  # MuJoCo 环境
                joint_names_for_ik=self.joint_names,  # 求解关节名
                body_name_trgt="tcp_link",  # 目标末端连杆
                q_init=q,  # 当前关节角初值
                p_trgt=self.p0,  # 更新后的目标位置
                R_trgt=self.R0,  # 更新后的目标姿态
                max_ik_tick=50,  # 最大 IK 迭代次数
                ik_stepsize=1.0,  # IK 迭代步长
                ik_eps=1e-2,  # 收敛误差阈值
                ik_th=np.radians(5.0),  # 单步最大关节角度变化阈值（5 度）
                render=False,  # 关闭 IK 求解过程渲染
                verbose_warning=False,  # 关闭未收敛警告打印
            )
        elif self.action_type == "delta_joint_angle":  # 关节角增量控制模式
            q = action[:-1] + self.last_q  # 在上一时刻关节角基础上叠加增量
        elif self.action_type == "joint_angle":  # 绝对关节角控制模式（用于策略推理与数据回放）
            q = action[:-1]  # 直接取前 6 维作为目标关节角
        else:
            raise ValueError("action_type not recognized")  # 未知动作类型抛出异常

        gripper_cmd = np.array([action[-1]] * 4)  # 将第 7 维夹爪指令复制给 4 个夹爪驱动关节
        gripper_cmd[[1, 3]] *= 0.8  # 第 2、4 个副指关节按 0.8 比例联动缩放
        self.compute_q = q  # 保存计算得到的 6 轴目标关节角
        q = np.concatenate([q, gripper_cmd])  # 拼接 6 轴臂关节与 4 轴夹爪指令为 10 维控制向量

        self.q = q  # 更新环境目标控制量（由 step_env() 写入执行器）
        if self.state_type == "joint_angle":  # 若要求返回关节状态
            return self.get_joint_state()
        elif self.state_type == "ee_pose":  # 若要求返回末端位姿
            return self.get_ee_pose()
        elif self.state_type == "delta_q":  # 若要求返回关节角增量
            dq = self.get_delta_q()
            return dq
        else:
            raise ValueError("state_type not recognized")  # 未知状态类型抛出异常

    def step_env(self):
        self.env.step(self.q)  # 将当前目标关节指令 self.q 传入 MuJoCo 并推进一个仿真步长

    def grab_image(self):
        """
        grab images from the environment
        returns:
            rgb_agent: np.array, rgb image from the agent's view
            rgb_ego: np.array, rgb image from the egocentric
        """
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name="agentview")  # 获取第三人称固定主视角 RGB 图像
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name="egocentric")  # 获取腕部第一人称视角 RGB 图像
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(cam_name="sideview")  # 获取侧视角 RGB 图像（用于遥操作辅助观察）
        return self.rgb_agent, self.rgb_ego  # 返回主视角与腕部视角图像

    def render(self, teleop=False):
        """
        Render the environment
        """
        self.env.plot_time()  # 在画面中显示当前仿真时间
        p_current, R_current = self.env.get_pR_body(body_name="tcp_link")  # 获取当前末端执行器中心位置与姿态
        R_current = R_current @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])  # 调整胶囊体朝向以匹配末端轴向
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95, 0.05, 0.05, 0.5])  # 在末端位置绘制半透明红色标记球
        self.env.plot_capsule(
            p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05, 0.95, 0.05, 0.5]  # 绘制半透明绿色胶囊体指示末端朝向
        )
        rgb_egocentric_view = add_title_to_img(
            self.rgb_ego, text="Egocentric View", shape=(640, 480)  # 为腕部图像添加标题并缩放到 640x480
        )
        rgb_agent_view = add_title_to_img(
            self.rgb_agent, text="Agent View", shape=(640, 480)  # 为主视角图像添加标题并缩放到 640x480
        )

        self.env.viewer_rgb_overlay(rgb_agent_view, loc="top right")  # 将主视角小窗叠加在右上角
        self.env.viewer_rgb_overlay(rgb_egocentric_view, loc="bottom right")  # 将腕部视角小窗叠加在右下角
        if teleop:  # 遥操作模式下额外显示侧视角与当前按键状态
            rgb_side_view = add_title_to_img(
                self.rgb_side, text="Side View", shape=(640, 480)  # 为侧视角图像添加标题
            )
            self.env.viewer_rgb_overlay(rgb_side_view, loc="top left")  # 将侧视角小窗叠加在左上角
            self.env.viewer_text_overlay(
                text1="Key Pressed", text2="%s" % (self.env.get_key_pressed_list())  # 显示单次按下的按键列表
            )
            self.env.viewer_text_overlay(
                text1="Key Repeated", text2="%s" % (self.env.get_key_repeated_list())  # 显示长按连续触发的按键列表
            )
        self.env.render()  # 刷新 MuJoCo 主渲染窗口

    def get_joint_state(self):
        """
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        """
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)  # 读取 6 个机械臂关节当前角度
        gripper = self.env.get_qpos_joint("rh_r1")  # 读取夹爪主关节当前开合位置
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0  # 二值化夹爪状态：大于 0.5 视为闭合（1.0），否则张开（0.0）
        return np.concatenate([qpos, [gripper_cmd]], dtype=np.float32)  # 拼接返回 7 维关节+夹爪状态

    def teleop_robot(self):
        """
        Teleoperate the robot using keyboard
        returns:
            action: np.array, action to take
            done: bool, True if the user wants to reset the teleoperation

        Keys:
            ---------     -----------------------
               w       ->        backward
            s  a  d        left   forward   right
            ---------      -----------------------
            In x, y plane

            ---------
            R: Moving Up
            F: Moving Down
            ---------
            In z axis

            ---------
            Q: Tilt left
            E: Tilt right
            UP: Look Upward
            Down: Look Donward
            Right: Turn right
            Left: Turn left
            ---------
            For rotation

            ---------
            z: reset
            SPACEBAR: gripper open/close
            ---------


        """
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)  # 初始化三维位置增量 [dx, dy, dz]
        drot = np.eye(3)  # 初始化三维旋转增量矩阵为单位阵
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):  # 长按 S：沿 +X 方向移动（前）
            dpos += np.array([0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):  # 长按 W：沿 -X 方向移动（后）
            dpos += np.array([-0.007, 0.0, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):  # 长按 A：沿 -Y 方向移动（左）
            dpos += np.array([0.0, -0.007, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):  # 长按 D：沿 +Y 方向移动（右）
            dpos += np.array([0.0, 0.007, 0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):  # 长按 R：沿 +Z 方向上升
            dpos += np.array([0.0, 0.0, 0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):  # 长按 F：沿 -Z 方向下降
            dpos += np.array([0.0, 0.0, -0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):  # 左方向键：绕 Y 轴正向旋转
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):  # 右方向键：绕 Y 轴负向旋转
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):  # 下方向键：绕 X 轴正向旋转
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):  # 上方向键：绕 X 轴负向旋转
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):  # 长按 Q：绕 Z 轴正向倾斜/旋转
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):  # 长按 E：绕 Z 轴负向倾斜/旋转
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):  # 单击 Z 键：返回零动作与重置信号 True
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):  # 单击空格键：切换夹爪开合状态
            self.gripper_state = not self.gripper_state
        drot = r2rpy(drot)  # 将旋转增量矩阵转换为欧拉角 [droll, dpitch, dyaw]
        action = np.concatenate(
            [dpos, drot, np.array([self.gripper_state], dtype=np.float32)],  # 拼接 7 维遥操作动作向量
            dtype=np.float32,
        )
        return action, False  # 返回动作向量与非重置信号 False

    def get_delta_q(self):
        """
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        """
        delta = self.compute_q - self.last_q  # 计算目标关节角相对上一帧的增量
        self.last_q = copy.deepcopy(self.compute_q)  # 更新上一帧关节角缓存
        gripper = self.env.get_qpos_joint("rh_r1")  # 读取夹爪主关节位置
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0  # 二值化夹爪状态
        return np.concatenate([delta, [gripper_cmd]], dtype=np.float32)  # 返回 6 维关节增量 + 1 维夹爪状态

    def check_success(self):
        """
        ['body_obj_mug_5', 'body_obj_plate_11']
        Check if the mug is stably standing upright on the plate
        + Mug's local +z axis should stay aligned with world +z (i.e. not lying on its side)
        + Mug should be at rest (small linear / angular velocity)
        + Gripper should be open and move upward above 0.9
        """
        p_mug = self.env.get_p_body("body_obj_mug_5")  # 获取杯子的世界坐标 [x, y, z]
        p_plate = self.env.get_p_body("body_obj_plate_11")  # 获取盘子的世界坐标 [x, y, z]
        R_mug = self.env.get_R_body("body_obj_mug_5")  # 获取杯子的 3x3 旋转矩阵

        # 杯身竖直：杯体局部 +z 轴与世界 +z 轴夹角小于 15°
        upright = R_mug[2, 2] > np.cos(np.deg2rad(15.0))  # 旋转矩阵 [2,2] 即局部 z 轴在世界 z 轴上的投影余弦值

        # 杯子已静止：读取自由关节的线速度与角速度
        jntadr = self.env.model.body("body_obj_mug_5").jntadr[0]  # 获取杯子自由关节（freejoint）的关节索引
        dofadr = self.env.model.jnt_dofadr[jntadr]  # 获取该自由关节在全局速度向量 qvel 中的起始偏移地址
        vel = self.env.data.qvel[dofadr : dofadr + 6]  # 切片取出杯子的 6 维速度（前 3 维线速度 + 后 3 维角速度）
        at_rest = np.linalg.norm(vel[:3]) < 0.05 and np.linalg.norm(vel[3:]) < 0.5  # 判断线速度 < 0.05 m/s 且角速度 < 0.5 rad/s

        if (
            upright  # 1. 杯子保持直立
            and at_rest  # 2. 杯子基本静止
            and np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.1  # 3. 杯子与盘子水平距离小于 0.1m
            and np.linalg.norm(p_mug[2] - p_plate[2]) < 0.6  # 4. 杯子与盘子高度差小于 0.6m
            and self.env.get_qpos_joint("rh_r1")[0] < 0.1  # 5. 夹爪已完全松开（关节角 < 0.1）
        ):
            p = self.env.get_p_body("tcp_link")[2]  # 读取当前机械臂末端高度 z
            if p > 0.9:  # 6. 末端已抬升到 0.9m 以上安全高度
                return True  # 所有条件满足，判定任务成功
        return False  # 任一条件未满足则返回 False

    def get_obj_pose(self):
        """
        returns:
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        """
        p_mug = self.env.get_p_body("body_obj_mug_5")  # 获取杯子位置
        p_plate = self.env.get_p_body("body_obj_plate_11")  # 获取盘子位置
        return p_mug, p_plate  # 返回杯子与盘子坐标

    def set_obj_pose(self, p_mug, p_plate):
        """
        Set the object poses
        args:
            p_mug: np.array, position of the mug
            p_plate: np.array, position of the plate
        """
        self.env.set_p_base_body(body_name="body_obj_mug_5", p=p_mug)  # 设置杯子基准位置
        self.env.set_R_base_body(body_name="body_obj_mug_5", R=np.eye(3, 3))  # 重置杯子朝向为单位阵
        self.env.set_p_base_body(body_name="body_obj_plate_11", p=p_plate)  # 设置盘子基准位置
        self.env.set_R_base_body(body_name="body_obj_plate_11", R=np.eye(3, 3))  # 重置盘子朝向为单位阵
        self.step_env()  # 推进一次仿真使位姿更新生效

    def get_ee_pose(self):
        """
        get the end effector pose of the robot + gripper state
        """
        p, R = self.env.get_pR_body(body_name="tcp_link")  # 获取末端执行器当前位置 p 与旋转矩阵 R
        rpy = r2rpy(R)  # 将旋转矩阵转为欧拉角 [roll, pitch, yaw]
        return np.concatenate([p, rpy], dtype=np.float32)  # 拼接返回 6 维末端位姿
