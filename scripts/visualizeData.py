"""
数据集轨迹 GUI 回放质检脚本：加载指定 Episode 并在 MuJoCo 窗口中重演动作与视角画面。
"""

import argparse

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from mujoco_env.y_env import SimpleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI 回放检查已采集的 LeRobot 数据集轨迹")
    parser.add_argument("--dataset-root", type=str, default="./dataset/smolvla_dataset", help="本地数据集路径")
    parser.add_argument("--repo-id", type=str, default="omy_sorting_vla", help="数据集标识名")
    parser.add_argument("--episode", type=int, default=0, help="要回放的 Episode 编号")
    parser.add_argument("--once", action="store_true", help="仅回放一遍即自动退出（默认循环播放）")
    return parser.parse_args()


def restore_episode_initial_state(env: SimpleEnv, first_frame: dict) -> None:
    """在第 0 帧恢复录制时的物体初始位置与语言指令。"""
    obj_init = first_frame["obj_init"][0].numpy()
    if len(obj_init) >= 9:
        env.set_obj_pose(obj_init[:3], obj_init[3:6], obj_init[6:9])
    else:
        env.set_obj_pose(obj_init[:3], obj_init[3:6])

    if "task" in first_frame:
        task_val = first_frame["task"]
        task_str = task_val[0] if isinstance(task_val, (list, tuple)) else str(task_val)
        env.set_instruction(given=task_str)


def main() -> None:
    args = parse_args()
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=[args.episode])
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    total_frames = len(dataset)

    env = SimpleEnv("./asset/example_scene_y.xml", action_type="joint_angle")
    env.reset()
    step = 0
    data_iter = iter(dataloader)

    try:
        while env.env.is_viewer_alive():
            env.step_env()
            if env.env.loop_every(HZ=20):
                data = next(data_iter)
                if step == 0:
                    restore_episode_initial_state(env, data)

                env.step(data["action"][0].numpy())
                rgb_agent = (data["observation.image"][0].numpy() * 255).astype(np.uint8)
                rgb_ego = (data["observation.wrist_image"][0].numpy() * 255).astype(np.uint8)
                env.rgb_agent = np.transpose(rgb_agent, (1, 2, 0))
                env.rgb_ego = np.transpose(rgb_ego, (1, 2, 0))
                env.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
                env.render(idx=args.episode)
                step += 1

                if step == total_frames:
                    print(f"[Replay] Episode {args.episode} ({total_frames} frames) | success={env.check_success()}")
                    if args.once:
                        break
                    data_iter = iter(dataloader)
                    env.reset()
                    step = 0
    finally:
        env.close()


if __name__ == "__main__":
    main()
