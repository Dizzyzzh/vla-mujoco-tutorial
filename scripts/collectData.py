"""
手柄 / 键盘双模遥操作数据采集脚本（LeRobot v3.0 格式，20Hz 控制频率）。

手柄防脏数据三招：
  1. 推摇杆/按键产生动作才自动触发开录（避免开头大量静止废帧）
  2. 按住 B 键切换 0.4x 慢速档精细对准水果
  3. 失误后按 Y / Start 键清空当前轨迹缓存并重新摆放物体

每帧写入的数据集字段与 Shape：
  - observation.image       : (256, 256, 3) uint8   主视角 RGB
  - observation.wrist_image : (256, 256, 3) uint8   腕部视角 RGB
  - observation.state       : (7,) float32          当前 6 轴关节角 + 夹爪
  - action                  : (7,) float32          目标 6 轴关节角 + 夹爪
  - obj_init                : (9,) float32          苹果/橙子/红牛罐初始坐标
  - task                    : str                   英文语言指令
"""

import argparse
from datetime import datetime

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from mujoco_env.y_env import SimpleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手柄/键盘遥操作采集 LeRobot 多任务跨障分拣数据集")
    parser.add_argument("--num-episodes", type=int, default=50, help="本次计划采集的成功轨迹条数")
    parser.add_argument("--task-mode", type=str, default="cycle", help="任务调度：cycle（四任务均衡）、random 或 0/1/2/3")
    parser.add_argument("--repo-id", type=str, default="omy_sorting_vla", help="数据集标识名")
    parser.add_argument("--output-dir", type=str, default=None, help="新数据集保存路径")
    parser.add_argument("--resume-dir", type=str, default=None, help="已有数据集路径（传此参数则在原数据上追加采集）")
    parser.add_argument("--ood-distractor", action="store_true", help="在抓取区额外放置红牛罐干扰物")
    parser.add_argument("--ood-position", action="store_true", help="扩大物体初始随机位置范围")
    return parser.parse_args()


def get_next_task_idx(ep_idx: int, mode: str) -> int | None:
    """根据全局轨迹序号与调度模式返回下一回合的任务索引。"""
    if mode in {"0", "1", "2", "3"}:
        return int(mode)
    if mode == "cycle":
        return ep_idx % len(SimpleEnv.TASKS)
    if mode == "random":
        return None
    raise ValueError(f"Unsupported --task-mode '{mode}', expected 'cycle', 'random', or '0'/'1'/'2'/'3'.")


def init_dataset(args: argparse.Namespace) -> LeRobotDataset:
    """新建或断点续录 LeRobot v3.0 数据集。"""
    if args.resume_dir is not None:
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=args.resume_dir)
        print(f"[Dataset] Resuming from {args.resume_dir} (existing={dataset.meta.total_episodes})")
        return dataset

    root = args.output_dir or f"./dataset/teleop_sorting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[Dataset] Creating new dataset at: {root}")
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        robot_type="omy",
        fps=20,
        features={
            "observation.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
            "observation.wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
            "observation.state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
            "obj_init": {"dtype": "float32", "shape": (9,), "names": ["obj_init"]},
        },
    )


def main() -> None:
    args = parse_args()
    dataset = init_dataset(args)
    start_offset = 0 if args.resume_dir is None else int(dataset.meta.total_episodes)

    env = SimpleEnv(
        "./asset/example_scene_y.xml",
        action_type="eef_pose",
        task_idx=get_next_task_idx(start_offset, args.task_mode),
        ood_distractor=args.ood_distractor,
        ood_position=args.ood_position,
    )

    episode_id = 0
    record_flag = False
    try:
        while env.env.is_viewer_alive() and episode_id < args.num_episodes:
            env.step_env()
            if env.env.loop_every(HZ=20):
                if env.check_success():
                    if dataset.has_pending_frames():
                        dataset.save_episode()
                        episode_id += 1
                        print(f"Saved episode {episode_id}/{args.num_episodes} | Task: {env.instruction}")
                    env.reset(
                        task_idx=get_next_task_idx(start_offset + episode_id, args.task_mode),
                        ood_distractor=args.ood_distractor,
                        ood_position=args.ood_position,
                    )
                    record_flag = False
                else:
                    teleop_action, reset = env.teleop_robot()
                    if not record_flag and np.any(np.abs(teleop_action) > 1e-6):
                        record_flag = True
                        print(f"Start recording ep {episode_id + 1}/{args.num_episodes} | {env.instruction}")

                    if reset:
                        if dataset.has_pending_frames():
                            dataset.clear_episode_buffer()
                        env.reset(
                            task_idx=get_next_task_idx(start_offset + episode_id, args.task_mode),
                            ood_distractor=args.ood_distractor,
                            ood_position=args.ood_position,
                        )
                        record_flag = False
                        continue

                    obs_state = np.asarray(env.get_joint_state(), dtype=np.float32)
                    agent_img, wrist_img = env.grab_image()
                    agent_img = np.asarray(Image.fromarray(agent_img).resize((256, 256)), dtype=np.uint8)
                    wrist_img = np.asarray(Image.fromarray(wrist_img).resize((256, 256)), dtype=np.uint8)

                    env.step(teleop_action)
                    if record_flag:
                        dataset.add_frame(
                            {
                                "observation.image": agent_img,
                                "observation.wrist_image": wrist_img,
                                "observation.state": obs_state,
                                "action": np.asarray(env.q[:7], dtype=np.float32),
                                "obj_init": np.asarray(env.obj_init_pose, dtype=np.float32),
                                "task": env.instruction,
                            }
                        )

                env.render(teleop=True, idx=episode_id)
    finally:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        dataset.finalize()
        env.close()

    print(f"Dataset saved to: {dataset.root}")


if __name__ == "__main__":
    main()
