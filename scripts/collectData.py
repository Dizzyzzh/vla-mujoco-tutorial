import argparse
from datetime import datetime

import lerobot
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mujoco_env.y_env import SimpleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于键盘 + 游戏手柄双模遥操作的 LeRobot 多任务跨障分拣演示数据采集脚本"
    )
    parser.add_argument("--num-episodes", type=int, default=50, help="本次计划采集的成功演示轨迹条数")
    parser.add_argument(
        "--task-mode",
        type=str,
        default="cycle",
        help="任务指令调度模式：cycle（四任务均衡轮询）、random（随机）或 0/1/2/3（固定单任务）",
    )
    parser.add_argument("--repo-id", type=str, default="omy_sorting_vla", help="数据集仓库标识名")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="新数据集保存目录（默认 ./dataset/teleop_sorting_YYYYMMDD_HHMMSS）",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="已有数据集目录，设置后将在该目录上断点续录追加轨迹",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default="./asset/example_scene_y.xml",
        help="MuJoCo 跨障分拣场景主 XML 文件路径",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（默认 None 每回合随机物体初位）")
    parser.add_argument("--ood-distractor", action="store_true", help="在抓取区额外放入红牛罐干扰物")
    parser.add_argument("--ood-position", action="store_true", help="扩大抓取区物体位置随机采样范围")
    parser.add_argument(
        "--image-dtype",
        type=str,
        choices=["image", "video"],
        default="image",
        help="视觉模态存储格式（默认 image）",
    )
    return parser.parse_args()


def get_next_task_idx(ep_idx: int, mode: str) -> int | None:
    """根据当前已采集的全局轨迹序号 ep_idx 与调度模式 mode，返回下一条 episode 的任务索引。"""
    if mode in {"0", "1", "2", "3"}:
        return int(mode)
    if mode == "cycle":
        return ep_idx % len(SimpleEnv.TASKS)
    if mode == "random":
        return None
    raise ValueError(
        f"Unsupported --task-mode '{mode}', expected 'cycle', 'random', or '0'/'1'/'2'/'3'."
    )


def init_dataset(args: argparse.Namespace) -> LeRobotDataset:
    """初始化新建或断点续录的 LeRobot v3.0 数据集，Schema 与 auto_collect.py 完全一致。"""
    if args.resume_dir is not None:
        print(f"[Dataset] Resuming dataset from: {args.resume_dir}")
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=args.resume_dir)
        print(f"[Dataset] Existing episodes: {dataset.meta.total_episodes}")
        return dataset

    root = (
        args.output_dir
        if args.output_dir is not None
        else f"./dataset/teleop_sorting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"[Dataset] Creating new dataset at: {root} (image_dtype={args.image_dtype})")
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        robot_type="omy",
        fps=20,
        features={
            "observation.image": {
                "dtype": args.image_dtype,
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": args.image_dtype,
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["obj_init"],
            },
        },
    )


def main() -> None:
    args = parse_args()
    print(f"lerobot version: {lerobot.__version__}")

    dataset = init_dataset(args)
    start_ep_offset = 0 if args.resume_dir is None else int(dataset.meta.total_episodes)
    cur_task_idx = get_next_task_idx(start_ep_offset, args.task_mode)

    env = SimpleEnv(
        args.xml_path,
        action_type="eef_pose",
        seed=args.seed,
        task_idx=cur_task_idx,
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
                    else:
                        print("Success detected before any frame was recorded, skip saving.")

                    next_task_idx = get_next_task_idx(start_ep_offset + episode_id, args.task_mode)
                    env.reset(
                        seed=args.seed,
                        task_idx=next_task_idx,
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
                        env.reset(
                            seed=args.seed,
                            task_idx=get_next_task_idx(start_ep_offset + episode_id, args.task_mode),
                            ood_distractor=args.ood_distractor,
                            ood_position=args.ood_position,
                        )
                        if dataset.has_pending_frames():
                            dataset.clear_episode_buffer()
                        record_flag = False
                        continue

                    obs_joint_state = np.asarray(env.get_joint_state(), dtype=np.float32)
                    agent_image, wrist_image = env.grab_image()
                    agent_image = np.asarray(
                        Image.fromarray(agent_image).resize((256, 256)), dtype=np.uint8
                    )
                    wrist_image = np.asarray(
                        Image.fromarray(wrist_image).resize((256, 256)), dtype=np.uint8
                    )

                    env.step(teleop_action)
                    target_action = np.asarray(env.q[:7], dtype=np.float32)

                    if record_flag:
                        dataset.add_frame(
                            {
                                "observation.image": agent_image,
                                "observation.wrist_image": wrist_image,
                                "observation.state": obs_joint_state,
                                "action": target_action,
                                "obj_init": np.asarray(env.obj_init_pose, dtype=np.float32),
                                "task": env.instruction,
                            }
                        )

                env.render(teleop=True, idx=episode_id)
    finally:
        if dataset.has_pending_frames():
            print("Discard unfinished episode.")
            dataset.clear_episode_buffer()
        dataset.finalize()
        env.close()

    print(f"Dataset saved to: {dataset.root}")


if __name__ == "__main__":
    main()
