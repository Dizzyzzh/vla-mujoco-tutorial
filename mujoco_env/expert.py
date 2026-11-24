from dataclasses import dataclass
from typing import Optional

import numpy as np

from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r

# 末端夹爪垂直朝向桌面的基准旋转矩阵（与 SimpleEnv.reset 保持完全一致）
R_BASE = rpy2r(np.deg2rad([90.0, 0.0, 90.0]))


@dataclass
class ExpertConfig:
    """特权状态启发式专家（IK + 贝塞尔越障轨迹）超参数配置。"""

    control_hz: int = 20  # 控制与采样频率（Hz）
    randomize: bool = True  # 是否对抓取偏置、越障高度、速度与保持步数做轻量随机化

    # 抓取阶段参数
    grasp_xy_jitter: float = 0.004  # 抓取 XY 中心微调抖动上限（米）
    grasp_z_offset_range: tuple[float, float] = (0.002, 0.008)  # 相对物体几何中心的 Z 偏移（米）
    pre_grasp_z_range: tuple[float, float] = (0.940, 0.970)  # 预抓取悬停高度区间（米）
    yaw_jitter_rad: float = float(np.deg2rad(8.0))  # 安全抓取偏航角随机微调幅度（弧度）

    # 抬升与贝塞尔跨障参数
    lift_z_range: tuple[float, float] = (0.940, 0.975)  # 抓起后抬升高度区间（米）
    apex_z_range: tuple[float, float] = (0.965, 1.010)  # 越过隔板中点的贝塞尔最高点高度区间（米）
    pre_place_z_range: tuple[float, float] = (0.935, 0.965)  # 到达托盘上方的预放置高度区间（米）

    # 放置与撤离参数
    place_z_range: tuple[float, float] = (0.852, 0.866)  # 松爪放置高度区间（米，贴近盘底防弹飞）
    place_xy_jitter: float = 0.015  # 目标托盘内落点 XY 随机偏移上限（米）
    retreat_z_range: tuple[float, float] = (0.940, 0.965)  # 释放后撤离安全高度区间（需 > 0.90m）

    # 运动速度与时序保持参数
    speed_move_range: tuple[float, float] = (0.14, 0.20)  # 水平/抬升移动速度区间（m/s）
    speed_descend_range: tuple[float, float] = (0.10, 0.14)  # 垂直下降接近/放置速度区间（m/s）
    close_hold_steps: tuple[int, int] = (10, 14)  # 闭合夹爪稳定接触保持步数（20Hz 下 0.5~0.7s）
    open_hold_steps: tuple[int, int] = (8, 11)  # 松开夹爪完全张开保持步数（20Hz 下 0.4~0.55s）
    max_settle_steps: int = 20  # 撤离到位后等待物体完全静止判定的最大保持步数

    # 工作空间安全限幅（与 SimpleEnv 保持一致）
    workspace_x: tuple[float, float] = (0.14, 0.46)
    workspace_y: tuple[float, float] = (-0.29, 0.29)
    workspace_z: tuple[float, float] = (0.822, 1.15)


class PrivilegedSortingExpert:
    """
    基于仿真特权状态（物体几何中心、隔板位置、逆运动学 IK 与二次贝塞尔曲线）的
    多任务跨障分拣自动演示专家。
    """

    def __init__(self, config: Optional[ExpertConfig] = None) -> None:
        self.cfg = config if config is not None else ExpertConfig()
        self._waypoints: list[tuple[np.ndarray, float, float, str]] = []
        self._step_idx: int = 0
        self._current_phase: str = "IDLE"

    @property
    def current_phase(self) -> str:
        """返回当前正在执行的运动阶段名称。"""
        return self._current_phase

    @property
    def total_planned_steps(self) -> int:
        """返回本条轨迹预规划的总步数。"""
        return len(self._waypoints)

    def _sample_uniform(
        self, rng: np.random.Generator, low_high: tuple[float, float]
    ) -> float:
        """若开启随机化则在区间内均匀采样，否则返回区间中点。"""
        low, high = low_high
        if not self.cfg.randomize:
            return 0.5 * (low + high)
        return float(rng.uniform(low, high))

    def _sample_int(
        self, rng: np.random.Generator, low_high: tuple[int, int]
    ) -> int:
        """若开启随机化则在闭区间内均匀采样整数步数，否则取中值。"""
        low, high = low_high
        if not self.cfg.randomize:
            return int(round(0.5 * (low + high)))
        return int(rng.integers(low, high + 1))

    def _clip_workspace(self, p: np.ndarray) -> np.ndarray:
        """将三维目标位置裁剪到安全工作空间包络内。"""
        return np.array(
            [
                np.clip(p[0], self.cfg.workspace_x[0], self.cfg.workspace_x[1]),
                np.clip(p[1], self.cfg.workspace_y[0], self.cfg.workspace_y[1]),
                np.clip(p[2], self.cfg.workspace_z[0], self.cfg.workspace_z[1]),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _ease_cos(u: float) -> float:
        """余弦平滑加减速时间映射 s(u) = 0.5 * (1 - cos(pi * u))，消除轨迹启停突变。"""
        return float(0.5 * (1.0 - np.cos(np.pi * np.clip(u, 0.0, 1.0))))

    @staticmethod
    def _angle_diff(target: float, source: float) -> float:
        """计算最短角度差 [-pi, pi]。"""
        return float((target - source + np.pi) % (2.0 * np.pi) - np.pi)

    def _select_grasp_yaw(
        self, env, target_xy: np.ndarray, rng: np.random.Generator
    ) -> float:
        """
        结合非目标物体、红牛罐干扰物及隔板几何位置，自动筛选无碰撞的夹爪抓取偏航角。
        """
        obstacles: list[tuple[np.ndarray, float]] = []

        other_center = env.get_obj_geom_center(env.obj_other)
        obstacles.append((other_center[:2], 0.032))

        redbull_center = env.get_obj_geom_center("body_obj_redbull")
        if redbull_center[0] > -0.5:  # 确认红牛罐在桌面上而非场外停放区
            obstacles.append((redbull_center[:2], 0.028))

        candidates = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 36, endpoint=False)
        best_yaw = 0.0
        best_score = -1e9
        best_clearance = -1e9

        finger_half_span = 0.058  # 张开状态下两指尖相对 tcp_link 中心的半宽距离
        for yaw in candidates:
            axis = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
            f_plus = target_xy + finger_half_span * axis
            f_minus = target_xy - finger_half_span * axis

            clearances = []
            for f_xy in (f_plus, f_minus):
                # 1) 到各障碍物边界的距离
                for obs_xy, obs_rad in obstacles:
                    clearances.append(float(np.linalg.norm(f_xy - obs_xy) - obs_rad))
                # 2) 到中间隔板（Y = 0）的安全距离
                clearances.append(float(abs(f_xy[1]) - 0.012))

            min_clear = min(clearances)
            score = min_clear - 0.015 * abs(float(yaw))
            if score > best_score:
                best_score = score
                best_yaw = float(yaw)
                best_clearance = min_clear

        if self.cfg.randomize and best_clearance > 0.035:
            best_yaw += float(rng.uniform(-self.cfg.yaw_jitter_rad, self.cfg.yaw_jitter_rad))

        return float(best_yaw)

    def _append_linear_segment(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        yaw_start: float,
        yaw_end: float,
        gripper_cmd: float,
        speed: float,
        phase_name: str,
        min_steps: int = 4,
    ) -> None:
        """按速度与余弦加减速插值生成直线段路点序列。"""
        dist = float(np.linalg.norm(p_end - p_start))
        n_steps = max(min_steps, int(np.ceil((dist / max(speed, 1e-4)) * self.cfg.control_hz)))
        dyaw = self._angle_diff(yaw_end, yaw_start)

        for k in range(1, n_steps + 1):
            s = self._ease_cos(k / n_steps)
            p_k = self._clip_workspace(p_start + s * (p_end - p_start))
            yaw_k = yaw_start + s * dyaw
            self._waypoints.append((p_k, yaw_k, gripper_cmd, phase_name))

    def _append_hold_segment(
        self,
        p_hold: np.ndarray,
        yaw_hold: float,
        gripper_cmd: float,
        n_steps: int,
        phase_name: str,
    ) -> None:
        """在指定位姿保持若干控制步（用于夹紧接触、松爪释放与静止等待）。"""
        p_clipped = self._clip_workspace(p_hold)
        for _ in range(n_steps):
            self._waypoints.append((p_clipped.copy(), yaw_hold, gripper_cmd, phase_name))

    def _append_bezier_segment(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        yaw_hold: float,
        gripper_cmd: float,
        speed: float,
        phase_name: str,
    ) -> None:
        """生成二次贝塞尔越障平滑弧线：B(s) = (1-s)^2 P0 + 2(1-s)s P1 + s^2 P2。"""
        # 用两段折线长度近似贝塞尔弧长以合理分配插值步数
        approx_len = float(np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1))
        n_steps = max(12, int(np.ceil((approx_len / max(speed, 1e-4)) * self.cfg.control_hz)))

        for k in range(1, n_steps + 1):
            s = self._ease_cos(k / n_steps)
            omt = 1.0 - s
            p_k = (omt * omt) * p0 + (2.0 * omt * s) * p1 + (s * s) * p2
            self._waypoints.append((self._clip_workspace(p_k), yaw_hold, gripper_cmd, phase_name))

    def reset(self, env, rng: Optional[np.random.Generator] = None) -> None:
        """根据当前仿真环境重置后的物体真实位姿与任务指令，预规划整条 20Hz 演示轨迹。"""
        if rng is None:
            rng = np.random.default_rng()

        self._waypoints = []
        self._step_idx = 0
        self._current_phase = "APPROACH_ABOVE"

        p_start = env.env.get_p_body("tcp_link").copy()
        obj_center = env.get_obj_geom_center(env.obj_target)
        tray_center = env.env.get_p_body(env.tray_target).copy()
        redbull_center = env.get_obj_geom_center("body_obj_redbull")
        distractor_active = bool(redbull_center[0] > -0.5)

        # 采样动作速度与各阶段高度参数
        speed_move = self._sample_uniform(rng, self.cfg.speed_move_range)
        speed_desc = self._sample_uniform(rng, self.cfg.speed_descend_range)

        z_pre_grasp = self._sample_uniform(rng, self.cfg.pre_grasp_z_range)
        z_lift = self._sample_uniform(rng, self.cfg.lift_z_range)
        if distractor_active:  # 桌上有 11.5cm 高的红牛罐时抬高净空（top ≈ 0.916m）
            z_pre_grasp = max(z_pre_grasp, 0.970)
            z_lift = max(z_lift, 0.970)

        z_grasp = float(obj_center[2]) + self._sample_uniform(rng, self.cfg.grasp_z_offset_range)
        z_apex = max(
            self._sample_uniform(rng, self.cfg.apex_z_range),
            z_lift + 0.015,
        )
        z_pre_place = self._sample_uniform(rng, self.cfg.pre_place_z_range)
        if distractor_active:
            z_pre_place = max(z_pre_place, 0.960)
        z_place = self._sample_uniform(rng, self.cfg.place_z_range)
        z_retreat = self._sample_uniform(rng, self.cfg.retreat_z_range)

        # 采样微小的抓取 XY 抖动与托盘落点 XY 抖动
        if self.cfg.randomize:
            grasp_dx = float(rng.uniform(-self.cfg.grasp_xy_jitter, self.cfg.grasp_xy_jitter))
            grasp_dy = float(rng.uniform(-self.cfg.grasp_xy_jitter, self.cfg.grasp_xy_jitter))
            place_dx = float(rng.uniform(-self.cfg.place_xy_jitter, self.cfg.place_xy_jitter))
            place_dy = float(rng.uniform(-self.cfg.place_xy_jitter, self.cfg.place_xy_jitter))
            mid_dx = float(rng.uniform(-0.015, 0.015))
            mid_dy = float(rng.uniform(-0.015, 0.015))
        else:
            grasp_dx = grasp_dy = place_dx = place_dy = mid_dx = mid_dy = 0.0

        grasp_xy = np.array([obj_center[0] + grasp_dx, obj_center[1] + grasp_dy], dtype=np.float64)
        place_xy = np.array([tray_center[0] + place_dx, tray_center[1] + place_dy], dtype=np.float64)
        yaw_grasp = self._select_grasp_yaw(env, grasp_xy, rng)

        p_pre_grasp = self._clip_workspace(np.array([grasp_xy[0], grasp_xy[1], z_pre_grasp]))
        p_grasp = self._clip_workspace(np.array([grasp_xy[0], grasp_xy[1], z_grasp]))
        p_lift = self._clip_workspace(np.array([grasp_xy[0], grasp_xy[1], z_lift]))
        p_pre_place = self._clip_workspace(np.array([place_xy[0], place_xy[1], z_pre_place]))
        p_place = self._clip_workspace(np.array([place_xy[0], place_xy[1], z_place]))
        p_retreat = self._clip_workspace(np.array([place_xy[0], place_xy[1], z_retreat]))

        # 计算二次贝塞尔中间控制点 P1，使曲线中点 s=0.5 精确达到期望顶点高度 z_apex
        p1_z = 2.0 * z_apex - 0.5 * (p_lift[2] + p_pre_place[2])
        p_bezier_mid = self._clip_workspace(
            np.array(
                [
                    0.5 * (p_lift[0] + p_pre_place[0]) + mid_dx,
                    0.0 + mid_dy,
                    p1_z,
                ],
                dtype=np.float64,
            )
        )

        # 拼接 8 段完整演示轨迹
        self._append_linear_segment(
            p_start, p_pre_grasp, 0.0, yaw_grasp, 0.0, speed_move, "APPROACH_ABOVE"
        )
        self._append_linear_segment(
            p_pre_grasp, p_grasp, yaw_grasp, yaw_grasp, 0.0, speed_desc, "DESCEND"
        )
        self._append_hold_segment(
            p_grasp,
            yaw_grasp,
            1.0,
            self._sample_int(rng, self.cfg.close_hold_steps),
            "GRASP_CLOSE",
        )
        self._append_linear_segment(
            p_grasp, p_lift, yaw_grasp, yaw_grasp, 1.0, speed_move, "LIFT"
        )
        self._append_bezier_segment(
            p_lift, p_bezier_mid, p_pre_place, yaw_grasp, 1.0, speed_move, "CROSS_BARRIER"
        )
        self._append_linear_segment(
            p_pre_place, p_place, yaw_grasp, yaw_grasp, 1.0, speed_desc, "DESCEND_PLACE"
        )
        self._append_hold_segment(
            p_place,
            yaw_grasp,
            0.0,
            self._sample_int(rng, self.cfg.open_hold_steps),
            "RELEASE",
        )
        self._append_linear_segment(
            p_place, p_retreat, yaw_grasp, yaw_grasp, 0.0, speed_move, "RETREAT"
        )
        self._append_hold_segment(
            p_retreat, yaw_grasp, 0.0, self.cfg.max_settle_steps, "SETTLE"
        )

    def get_action(self, env) -> tuple[np.ndarray, bool, str]:
        """
        计算当前步的 7 维绝对关节控制量 [j1..j6, gripper]，并执行在线抓空/IK 失败早停检测。
        返回: (action_7d, failed, fail_reason)
        """
        idx = min(self._step_idx, len(self._waypoints) - 1)
        p_des, yaw_des, gripper_cmd, phase_name = self._waypoints[idx]
        self._current_phase = phase_name
        self._step_idx += 1

        # 抬升完成并开始跨障时，检查目标水果是否真的随夹爪被抬离桌面
        if phase_name == "CROSS_BARRIER":
            obj_z = float(env.get_obj_geom_center(env.obj_target)[2])
            if obj_z < 0.840:
                return np.zeros(7, dtype=np.float32), True, "grasp_or_drop_fail"

        R_des = rpy2r(np.array([0.0, 0.0, yaw_des], dtype=np.float64)) @ R_BASE
        q_curr = env.env.get_qpos_joints(joint_names=env.joint_names)
        q_ik, ik_err_stack, _ = solve_ik(
            env=env.env,
            joint_names_for_ik=env.joint_names,
            body_name_trgt="tcp_link",
            q_init=q_curr,
            p_trgt=p_des,
            R_trgt=R_des,
            max_ik_tick=60,
            ik_stepsize=1.0,
            ik_eps=1e-2,
            ik_th=np.radians(5.0),
            restore_state=True,
            render=False,
            verbose_warning=False,
        )

        ik_err = float(np.linalg.norm(ik_err_stack))
        if ik_err > 0.05:
            return np.zeros(7, dtype=np.float32), True, "ik_fail"

        action_7d = np.concatenate([q_ik, [gripper_cmd]]).astype(np.float32)
        return action_7d, False, ""
