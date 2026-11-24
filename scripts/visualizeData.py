import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mujoco_env.y_env import SimpleEnv

try:
    from scripts.eval_benchmark import compose_dual_view_frame, save_episode_video
except ImportError:
    from eval_benchmark import compose_dual_view_frame, save_episode_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="回放并质检 LeRobot 数据集轨迹（支持交互式 GUI 窗口与无界面物理复现/视频导出）"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./dataset/smolvla_dataset",
        help="待回放质检的本地数据集根目录",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="omy_sorting_vla",
        help="数据集仓库标识名（需与采集时保持一致）",
    )
    parser.add_argument("--episode", type=int, default=0, help="指定回放的 episode 编号（从 0 开始）")
    parser.add_argument(
        "--xml-path",
        type=str,
        default="./asset/example_scene_y.xml",
        help="MuJoCo 跨障分拣场景主 XML 文件路径",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无界面离屏回放单轮轨迹并输出物理复现成功/碰撞校验结果",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="GUI 模式下仅回放单轮即自动退出（默认循环播放直到关闭窗口）",
    )
    parser.add_argument(
        "--save-video-dir",
        type=str,
        default=None,
        help="可选：将回放过程导出为双视角标注视频/GIF 的保存目录",
    )
    parser.add_argument(
        "--video-format",
        type=str,
        choices=["mp4", "gif", "both"],
        default="mp4",
        help="视频导出格式（mp4 / gif / both）",
    )
    return parser.parse_args()


def restore_episode_initial_state(env: SimpleEnv, first_frame: dict) -> None:
    """在回放第 0 帧恢复录制时的物体初位与语言指令。"""
    obj_init = first_frame["obj_init"][0].numpy()
    if len(obj_init) >= 9:
        env.set_obj_pose(obj_init[:3], obj_init[3:6], obj_init[6:9])
    else:
        env.set_obj_pose(obj_init[:3], obj_init[3:6])

    if "task" in first_frame:
        task_val = first_frame["task"]
        task_str = task_val[0] if isinstance(task_val, (list, tuple)) else str(task_val)
        env.set_instruction(given=task_str)


def run_headless_replay(args: argparse.Namespace, dataset: LeRobotDataset) -> None:
    """无界面精确按帧推进动作，验证数据集物理复现一致性并按需导出视频。"""
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    env = SimpleEnv(
        args.xml_path,
        action_type="joint_angle",
        headless=True,
        verbose=False,
    )
    substeps = int(round(1.0 / (env.env.dt * 20)))
    total_frames = len(dataset)
    barrier_hit = False
    video_frames: list[np.ndarray] = []

    try:
        env.reset()
        for step, data in enumerate(dataloader):
            if step == 0:
                restore_episode_initial_state(env, data)

            action = data["action"][0].numpy()
            env.step(action)
            for _ in range(substeps):
                env.step_env()
                if env.check_barrier_collision():
                    barrier_hit = True

            if args.save_video_dir is not None:
                agent_img, wrist_img = env.grab_image(include_side=False)
                status = "SUCCESS" if env.check_success() else "REPLAY"
                video_frames.append(
                    compose_dual_view_frame(
                        agent_img=agent_img,
                        wrist_img=wrist_img,
                        suite_name="replay",
                        ep_idx=args.episode,
                        step=step + 1,
                        max_steps=total_frames,
                        instruction=env.instruction,
                        status_text=status,
                        barrier_hit=barrier_hit,
                    )
                )

        replay_success = env.check_success()
        print(
            f"[Replay Check] episode={args.episode} | frames={total_frames} | "
            f"task='{env.instruction}' | replay_success={replay_success} | barrier_hit={barrier_hit}"
        )

        if args.save_video_dir is not None and video_frames:
            stem = f"replay_ep{args.episode:03d}_{'PASS' if replay_success else 'FAIL'}"
            saved = save_episode_video(
                video_frames, Path(args.save_video_dir), stem, args.video_format
            )
            print(f"[Video] Saved replay video: {', '.join(saved)}")
    finally:
        env.close()


def run_gui_replay(args: argparse.Namespace, dataset: LeRobotDataset) -> None:
    """交互式 MuJoCo 窗口循环回放数据集动作，并在右侧叠加数据集中保存的原始相机帧。"""
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    env = SimpleEnv(args.xml_path, action_type="joint_angle")
    total_frames = len(dataset)
    step = 0
    iter_dataloader = iter(dataloader)
    env.reset()

    try:
        while env.env.is_viewer_alive():
            env.step_env()
            if env.env.loop_every(HZ=20):
                data = next(iter_dataloader)
                if step == 0:
                    restore_episode_initial_state(env, data)

                action = data["action"][0].numpy()
                env.step(action)

                rgb_agent = (data["observation.image"][0].numpy() * 255).astype(np.uint8)
                rgb_ego = (data["observation.wrist_image"][0].numpy() * 255).astype(np.uint8)
                env.rgb_agent = np.transpose(rgb_agent, (1, 2, 0))
                env.rgb_ego = np.transpose(rgb_ego, (1, 2, 0))
                env.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
                env.render(idx=args.episode)
                step += 1

                if step == total_frames:
                    print(
                        f"[Replay] Finished episode {args.episode} ({total_frames} frames) | "
                        f"success={env.check_success()}"
                    )
                    if args.once:
                        break
                    iter_dataloader = iter(dataloader)
                    env.reset()
                    step = 0
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=[args.episode])
    if args.headless:
        run_headless_replay(args, dataset)
    else:
        run_gui_replay(args, dataset)


if __name__ == "__main__":
    main()
