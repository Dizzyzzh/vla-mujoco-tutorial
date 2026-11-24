import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import lerobot
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mujoco_env.expert import ExpertConfig, PrivilegedSortingExpert
from mujoco_env.y_env import SimpleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 MuJoCo 特权状态与 IK 贝塞尔专家的 LeRobot 多任务自动化数据生成脚本"
    )
    parser.add_argument("--num-episodes", type=int, default=200, help="计划采集的成功演示轨迹条数")
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
        default="./dataset/smolvla_dataset",
        help="新数据集保存根目录（默认 ./dataset/auto_sorting_YYYYMMDD_HHMMSS）",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="已有数据集目录，设置后将在该目录上继续追加轨迹",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default="./asset/example_scene_y.xml",
        help="MuJoCo 跨障分拣场景主 XML 文件路径",
    )
    parser.add_argument("--seed", type=int, default=42, help="主随机种子（控制环境布局与轨迹随机化）")
    parser.add_argument(
        "--seeds-file",
        type=str,
        default=None,
        help="可选：JSON 种子列表文件（支持 int 列表或含 seed/task_idx/ood_distractor/ood_position 的字典列表）",
    )
    parser.add_argument(
        "--distractor-prob",
        type=float,
        default=0.0,
        help="每回合启用红牛罐 OOD 干扰物的概率（0.0~1.0）",
    )
    parser.add_argument(
        "--ood-pos-prob",
        type=float,
        default=0.0,
        help="每回合启用空间外推 OOD 位置范围的概率（0.0~1.0）",
    )
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="关闭专家轨迹高度/速度/落点随机微扰，采用确定性中值轨迹",
    )
    parser.add_argument(
        "--allow-collision",
        action="store_true",
        help="允许保存曾触碰隔板的轨迹（默认严格拒收任何隔板碰撞轨迹）",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="开启 MuJoCo 可视化窗口观察采集过程（默认无界面离屏高速采集）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=260,
        help="单条轨迹最大控制步数（20Hz 下 260 步 = 13 秒）",
    )
    parser.add_argument(
        "--image-dtype",
        type=str,
        choices=["image", "video"],
        default="image",
        help="视觉模态存储格式（默认 image，与 collectData.py 保持一致）",
    )
    return parser.parse_args()


def resolve_task_idx(
    ep_index: int, task_mode: str, rng: np.random.Generator
) -> int:
    """根据调度模式与当前全局序号返回任务索引（0~3）。"""
    if task_mode == "cycle":
        return ep_index % len(SimpleEnv.TASKS)
    if task_mode == "random":
        return int(rng.integers(0, len(SimpleEnv.TASKS)))
    if task_mode in {"0", "1", "2", "3"}:
        return int(task_mode)
    raise ValueError(
        f"Unsupported --task-mode '{task_mode}', expected 'cycle', 'random', or '0'/'1'/'2'/'3'."
    )


def load_seed_specs(
    seeds_file: str,
    task_mode: str,
    distractor_prob: float,
    ood_pos_prob: float,
    rng: np.random.Generator,
) -> list[dict]:
    """加载外部种子配置文件，支持整数列表或带场景配置字典的列表。"""
    with open(seeds_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    if not isinstance(raw_items, list):
        raise ValueError(f"--seeds-file must contain a JSON list, got {type(raw_items).__name__}")

    specs: list[dict] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, int):
            specs.append(
                {
                    "seed": int(item),
                    "task_idx": resolve_task_idx(idx, task_mode, rng),
                    "ood_distractor": bool(rng.random() < distractor_prob),
                    "ood_position": bool(rng.random() < ood_pos_prob),
                }
            )
        elif isinstance(item, dict):
            specs.append(
                {
                    "seed": int(item.get("seed", int(rng.integers(0, 2**31 - 1)))),
                    "task_idx": int(
                        item["task_idx"]
                        if item.get("task_idx") is not None
                        else resolve_task_idx(idx, task_mode, rng)
                    ),
                    "ood_distractor": bool(
                        item["ood_distractor"]
                        if "ood_distractor" in item
                        else (rng.random() < distractor_prob)
                    ),
                    "ood_position": bool(
                        item["ood_position"]
                        if "ood_position" in item
                        else (rng.random() < ood_pos_prob)
                    ),
                }
            )
        else:
            raise ValueError(f"Invalid seed entry at index {idx}: {item!r}")
    return specs


def init_dataset(args: argparse.Namespace) -> LeRobotDataset:
    """初始化新建或断点续录的 LeRobot v3.0 数据集，Schema 与 collectData.py 完全一致。"""
    if args.resume_dir is not None:
        print(f"[Dataset] Resuming dataset from: {args.resume_dir}")
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=args.resume_dir)
        print(f"[Dataset] Existing episodes: {dataset.meta.total_episodes}")
        return dataset

    root = (
        args.output_dir
        if args.output_dir is not None
        else f"./dataset/auto_sorting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    master_rng = np.random.default_rng(args.seed)
    dataset = init_dataset(args)
    start_ep_offset = 0 if args.resume_dir is None else int(dataset.meta.total_episodes)

    seed_specs = None
    target_episodes = args.num_episodes
    if args.seeds_file is not None:
        seed_specs = load_seed_specs(
            args.seeds_file,
            args.task_mode,
            args.distractor_prob,
            args.ood_pos_prob,
            master_rng,
        )
        target_episodes = min(args.num_episodes, len(seed_specs))
        print(
            f"[Seeds] Loaded {len(seed_specs)} entries from {args.seeds_file}, "
            f"target_episodes={target_episodes}"
        )

    env = SimpleEnv(
        args.xml_path,
        action_type="joint_angle",
        seed=None,
        task_idx=0,
        headless=not args.gui,
        verbose=False,
    )
    expert = PrivilegedSortingExpert(
        ExpertConfig(randomize=not args.no_randomize)
    )

    substeps = int(round(1.0 / (env.env.dt * 20)))
    max_attempts = max(target_episodes * 3, target_episodes + 20)

    saved_episodes = 0
    attempts = 0
    seed_cursor = 0
    task_counts: Counter[int] = Counter()
    fail_counts: Counter[str] = Counter()
    episode_lengths: list[int] = []
    t_start = time.perf_counter()

    try:
        while saved_episodes < target_episodes and attempts < max_attempts:
            if args.gui and not env.env.is_viewer_alive():
                print("[Info] Viewer closed by user, stopping collection.")
                break

            if seed_specs is not None:
                if seed_cursor >= len(seed_specs):
                    break
                spec = seed_specs[seed_cursor]
                seed_cursor += 1
                ep_seed = spec["seed"]
                task_idx = spec["task_idx"]
                use_distractor = spec["ood_distractor"]
                use_ood_pos = spec["ood_position"]
            else:
                ep_seed = int(master_rng.integers(0, 2**31 - 1))
                task_idx = resolve_task_idx(
                    start_ep_offset + saved_episodes, args.task_mode, master_rng
                )
                use_distractor = bool(master_rng.random() < args.distractor_prob)
                use_ood_pos = bool(master_rng.random() < args.ood_pos_prob)

            attempts += 1
            ep_rng = np.random.default_rng(int(master_rng.integers(0, 2**31 - 1)))

            env.reset(
                seed=ep_seed,
                task_idx=task_idx,
                ood_distractor=use_distractor,
                ood_position=use_ood_pos,
            )
            expert.reset(env, ep_rng)

            frames_recorded = 0
            success = False
            hit_barrier = False
            fail_reason = ""
            viewer_closed = False

            for _ in range(args.max_steps):
                if frames_recorded > 0 and env.check_success():
                    success = True
                    break

                obs_state = np.asarray(env.get_joint_state(), dtype=np.float32)
                agent_img, wrist_img = env.grab_image(include_side=args.gui)
                agent_resized = np.asarray(
                    Image.fromarray(agent_img).resize((256, 256)), dtype=np.uint8
                )
                wrist_resized = np.asarray(
                    Image.fromarray(wrist_img).resize((256, 256)), dtype=np.uint8
                )

                target_action, failed, reason = expert.get_action(env)
                if failed:
                    fail_reason = reason
                    break

                env.step(target_action)
                dataset.add_frame(
                    {
                        "observation.image": agent_resized,
                        "observation.wrist_image": wrist_resized,
                        "observation.state": obs_state,
                        "action": target_action,
                        "obj_init": np.asarray(env.obj_init_pose, dtype=np.float32),
                        "task": env.instruction,
                    }
                )
                frames_recorded += 1

                for _ in range(substeps):
                    env.step_env()
                    if env.check_barrier_collision():
                        hit_barrier = True

                if hit_barrier and not args.allow_collision:
                    fail_reason = "barrier_collision"
                    break

                if args.gui:
                    if not env.env.is_viewer_alive():
                        viewer_closed = True
                        break
                    env.render(teleop=False, idx=saved_episodes)

            if viewer_closed:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                print("[Info] Viewer closed mid-episode, discarding unfinished buffer.")
                break

            valid_episode = (
                success
                and frames_recorded > 0
                and (not hit_barrier or args.allow_collision)
            )
            if valid_episode:
                dataset.save_episode()
                saved_episodes += 1
                task_counts[task_idx] += 1
                episode_lengths.append(frames_recorded)
                print(
                    f"[Saved {saved_episodes}/{target_episodes}] "
                    f"attempt={attempts} | task={task_idx} | frames={frames_recorded} | "
                    f"dist={use_distractor} | ood_pos={use_ood_pos} | barrier_hit={hit_barrier}"
                )
            else:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                if not fail_reason:
                    fail_reason = "barrier_collision" if hit_barrier else "timeout"
                fail_counts[fail_reason] += 1
                print(
                    f"[Skip] attempt={attempts} | seed={ep_seed} | task={task_idx} | "
                    f"reason={fail_reason} | phase={expert.current_phase} | frames={frames_recorded}"
                )
    finally:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        dataset.finalize()
        env.close()

    elapsed = time.perf_counter() - t_start
    attempt_succ_rate = (saved_episodes / attempts * 100.0) if attempts > 0 else 0.0
    eps_per_sec = (saved_episodes / elapsed) if elapsed > 0 else 0.0

    print("\n================ Auto-Collect Summary ================")
    print(f"Dataset root   : {dataset.root}")
    print(f"Saved episodes : {saved_episodes}/{target_episodes} (attempts={attempts}, rate={attempt_succ_rate:.1f}%)")
    if episode_lengths:
        print(
            f"Episode length : avg={np.mean(episode_lengths):.1f}, "
            f"min={min(episode_lengths)}, max={max(episode_lengths)} frames"
        )
    print("Per-task count :")
    for t_idx, spec in enumerate(SimpleEnv.TASKS):
        print(f"  Task {t_idx}: {task_counts[t_idx]:3d} | {spec['instruction']}")
    if fail_counts:
        print(f"Failure counts : {dict(fail_counts)}")
    else:
        print("Failure counts : none")
    print(f"Elapsed time   : {elapsed:.2f}s ({eps_per_sec:.2f} saved ep/s)")
    print("======================================================")


if __name__ == "__main__":
    main()
