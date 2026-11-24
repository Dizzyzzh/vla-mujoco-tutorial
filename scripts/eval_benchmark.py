import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from mujoco_env.expert import ExpertConfig, PrivilegedSortingExpert
from mujoco_env.y_env import SimpleEnv

# 未见过的同义改写指令集（与 SimpleEnv.set_instruction 关键字解析规则严格对齐）
LANG_OOD_INSTRUCTIONS = [
    "Grab the red fruit and drop it into the far container.",  # Task 0: apple -> blue tray
    "Move the red fruit over the wall into the near tray.",  # Task 1: apple -> green tray
    "Take the citrus fruit and put it in the far container.",  # Task 2: orange -> blue tray
    "Grab the citrus fruit and place it into the near tray.",  # Task 3: orange -> green tray
]

# 4 维度闭环评测套件配置
SUITES = {
    "id": {"ood_distractor": False, "ood_position": False, "lang_ood": False},
    "visual_ood": {"ood_distractor": True, "ood_position": False, "lang_ood": False},
    "spatial_ood": {"ood_distractor": False, "ood_position": True, "lang_ood": False},
    "lang_ood": {"ood_distractor": False, "ood_position": False, "lang_ood": True},
}

FAILURE_REASONS = [
    "wrong_object",
    "distractor_hijack",
    "wrong_tray",
    "release_or_retreat_fail",
    "barrier_drop",
    "transit_drop",
    "grasp_fail",
    "reach_timeout",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一多模型（ACT / SmolVLA / PI0.5 / Expert）无界面闭环评测与失败归因基准"
    )
    parser.add_argument("--ckpt-path", type=str, default=None, help="待评测本地策略权重目录")
    parser.add_argument(
        "--policy-type",
        type=str,
        choices=["auto", "act", "smolvla", "pi05", "expert"],
        default="auto",
        help="策略类型（auto 自动读取 config.json；expert 运行特权专家 Oracle 上限验证）",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default="./asset/example_scene_y.xml",
        help="MuJoCo 跨障分拣场景主 XML 文件路径",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="all",
        help="评测套件列表：all 或逗号分隔子集（如 id,visual_ood,spatial_ood,lang_ood）",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        default="cycle",
        help="任务调度模式：cycle（四任务均衡轮询）、random 或 0/1/2/3（固定单任务）",
    )
    parser.add_argument("--episodes-per-suite", type=int, default=20, help="每个评测套件连续运行回合数")
    parser.add_argument("--max-steps", type=int, default=400, help="单回合最大控制步数（20Hz 下 400 步 = 20 秒）")
    parser.add_argument("--seed", type=int, default=100, help="基准起始随机种子（不同模型同套种子公平对齐初位）")
    parser.add_argument("--device", type=str, default=None, help="推理计算设备（默认自动选 cuda / cpu）")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="推理精度覆盖（CUDA 下启用 autocast 或配置 PI0.5 权重精度）",
    )
    parser.add_argument("--n-action-steps", type=int, default=None, help="可选：覆盖策略每次执行的动作块步数")
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="可选：为 ACT 启用时序集成平滑（自动设 n_action_steps=1）",
    )
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=None,
        help="可选：覆盖 Flow Matching 去噪步数（SmolVLA -> num_steps；PI0.5 -> num_inference_steps）",
    )
    parser.add_argument("--compile", action="store_true", help="对策略前向计算启用 torch.compile 加速")
    parser.add_argument("--gui", action="store_true", help="弹出 MuJoCo 可视化窗口实时观看评测过程")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="评测输出目录（保存 summary.json、report.md 与 failed_seeds.json）",
    )
    parser.add_argument("--save-video-dir", type=str, default=None, help="双视角对比视频/GIF 保存目录")
    parser.add_argument(
        "--video-format",
        type=str,
        choices=["mp4", "gif", "both"],
        default="mp4",
        help="录制导出格式（mp4 / gif / both）",
    )
    parser.add_argument("--max-videos-per-suite", type=int, default=2, help="每个套件最多录制的代表性回合数")
    return parser.parse_args()


def parse_suites(suites_arg: str) -> list[str]:
    """解析评测套件名称列表，支持 all 或逗号分隔组合。"""
    if suites_arg.strip().lower() == "all":
        return list(SUITES.keys())
    selected = [s.strip() for s in suites_arg.split(",") if s.strip()]
    if not selected:
        raise ValueError("No valid suite specified in --suites.")
    for name in selected:
        if name not in SUITES:
            raise ValueError(f"Unknown suite '{name}'. Valid options: {list(SUITES.keys())}")
    return selected


def resolve_task_idx(ep_idx: int, task_mode: str, rng: np.random.Generator) -> int:
    """根据任务调度模式返回当前回合的任务索引（0~3）。"""
    if task_mode == "cycle":
        return ep_idx % len(SimpleEnv.TASKS)
    if task_mode == "random":
        return int(rng.integers(0, len(SimpleEnv.TASKS)))
    if task_mode in {"0", "1", "2", "3"}:
        return int(task_mode)
    raise ValueError(
        f"Unsupported --task-mode '{task_mode}', expected 'cycle', 'random', or '0'/'1'/'2'/'3'."
    )


def detect_policy_type(ckpt_path: str) -> str:
    """从本地 checkpoint 的 config.json 自动识别策略类型。"""
    cfg_file = Path(ckpt_path) / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Cannot auto-detect policy type: {cfg_file} does not exist.")
    with open(cfg_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_type = str(data.get("type", "")).lower().strip()
    if raw_type == "pi0.5":
        raw_type = "pi05"
    if raw_type not in {"act", "smolvla", "pi05"}:
        raise ValueError(f"Unsupported policy type '{raw_type}' in {cfg_file}")
    return raw_type


def load_policy_and_processors(
    policy_type: str, ckpt_path: str, device: str, args: argparse.Namespace
):
    """统一加载 ACT / SmolVLA / PI0.5 模型权重与配套 pre/post processor。"""
    from lerobot.policies.factory import make_pre_post_processors

    if policy_type == "act":
        from lerobot.policies.act.configuration_act import ACTConfig as ConfigCls
        from lerobot.policies.act.modeling_act import ACTPolicy as PolicyCls
    elif policy_type == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import (
            SmolVLAConfig as ConfigCls,
        )
        from lerobot.policies.smolvla.modeling_smolvla import (
            SmolVLAPolicy as PolicyCls,
        )
    elif policy_type == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config as ConfigCls
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as PolicyCls
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    cfg = ConfigCls.from_pretrained(ckpt_path)
    cfg.device = device

    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps

    if policy_type == "act" and args.temporal_ensemble_coeff is not None:
        cfg.n_action_steps = 1
        cfg.temporal_ensemble_coeff = args.temporal_ensemble_coeff

    if policy_type == "smolvla" and args.num_denoising_steps is not None:
        cfg.num_steps = args.num_denoising_steps
    elif policy_type == "pi05" and args.num_denoising_steps is not None:
        cfg.num_inference_steps = args.num_denoising_steps

    if policy_type == "pi05" and args.dtype != "auto":
        cfg.dtype = getattr(torch, args.dtype)

    policy = PolicyCls.from_pretrained(ckpt_path, config=cfg)
    policy.to(device)
    policy.eval()

    if args.compile:
        policy = torch.compile(policy)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def build_observation(
    env: SimpleEnv,
    policy_type: str,
    policy,
    img_transform,
    device: str,
    include_side: bool = False,
):
    """采集当前关节状态、相机画面与语言指令，组装标准 LeRobot 推理输入字典。"""
    state = env.get_joint_state()
    agent_img, wrist_img = env.grab_image(include_side=include_side)
    agent_tensor = img_transform(Image.fromarray(agent_img).resize((256, 256)))

    data = {
        "observation.state": torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0),
        "observation.image": agent_tensor.unsqueeze(0).to(device),
    }

    if "observation.wrist_image" in policy.config.input_features:
        wrist_tensor = img_transform(Image.fromarray(wrist_img).resize((256, 256)))
        data["observation.wrist_image"] = wrist_tensor.unsqueeze(0).to(device)

    # SmolVLA 与 PI0.5 的分词预处理器均要求 task 为单元素字符串列表
    if policy_type in ("smolvla", "pi05"):
        data["task"] = [env.instruction]

    return data, agent_img, wrist_img


def pop_cached_action(policy, policy_type: str):
    """
    当动作块队列非空且策略无需逐帧更新历史观测窗口时，直接弹出下一缓存动作，
    避免无界面批测在队列命中步重复执行离屏渲染与分词（与完整调用动作输出完全等价）。
    """
    raw_policy = getattr(policy, "_orig_mod", policy)
    cfg = raw_policy.config
    if policy_type == "smolvla":
        if (
            getattr(cfg, "n_obs_steps", 1) == 1
            and not raw_policy._rtc_enabled()
            and len(raw_policy._queues.get("action", ())) > 0
        ):
            return raw_policy._queues["action"].popleft()
    elif policy_type == "act":
        if (
            getattr(cfg, "temporal_ensemble_coeff", None) is None
            and len(raw_policy._action_queue) > 0
        ):
            return raw_policy._action_queue.popleft()
    elif policy_type == "pi05":
        if (
            not getattr(cfg, "use_visual_memory", False)
            and not getattr(cfg, "use_proprioceptive_memory", False)
            and not raw_policy._rtc_enabled()
            and len(raw_policy._action_queue) > 0
        ):
            return raw_policy._action_queue.popleft()
    return None


def compose_dual_view_frame(
    agent_img: np.ndarray,
    wrist_img: np.ndarray,
    suite_name: str,
    ep_idx: int,
    step: int,
    max_steps: int,
    instruction: str,
    status_text: str,
    barrier_hit: bool,
) -> np.ndarray:
    """将主视角与腕部视角拼接为带实验标注横幅的 640x288 RGB 帧。"""
    canvas = np.zeros((288, 640, 3), dtype=np.uint8)
    canvas[:48, :] = (28, 34, 44)  # 深灰蓝顶部状态横幅

    agent_panel = cv2.resize(agent_img, (320, 240), interpolation=cv2.INTER_AREA)
    wrist_panel = cv2.resize(wrist_img, (320, 240), interpolation=cv2.INTER_AREA)
    canvas[48:, :320] = agent_panel
    canvas[48:, 320:] = wrist_panel

    # 顶部第一行：套件、回合、步数与当前状态
    header_line = f"[{suite_name}] ep {ep_idx + 1} | step {step}/{max_steps} | {status_text}"
    status_color = (110, 235, 130) if status_text == "SUCCESS" else (240, 244, 248)
    cv2.putText(
        canvas, header_line, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA
    )

    # 顶部第二行：当前自然语言任务指令
    instr_disp = instruction if len(instruction) <= 72 else instruction[:69] + "..."
    cv2.putText(
        canvas,
        f"Task: {instr_disp}",
        (10, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 212, 226),
        1,
        cv2.LINE_AA,
    )

    # 若发生隔板碰撞则在右上角贴出醒目红色警示徽章
    if barrier_hit:
        cv2.rectangle(canvas, (520, 6), (632, 26), (210, 45, 45), -1)
        cv2.putText(
            canvas, "BARRIER HIT", (526, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
        )

    # 底部左右画面角标
    cv2.rectangle(canvas, (6, 54), (104, 74), (20, 24, 30), -1)
    cv2.putText(
        canvas, "Agent View", (11, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.rectangle(canvas, (326, 54), (420, 74), (20, 24, 30), -1)
    cv2.putText(
        canvas, "Wrist View", (331, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
    )
    return canvas


def save_episode_video(
    frames: list[np.ndarray],
    out_dir: Path,
    file_stem: str,
    video_format: str,
) -> list[str]:
    """将拼接后的 RGB 帧序列导出为 MP4 和/或 GIF 文件。"""
    if not frames:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    if video_format in ("mp4", "both"):
        mp4_path = out_dir / f"{file_stem}.mp4"
        writer = cv2.VideoWriter(
            str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 288)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open cv2.VideoWriter for {mp4_path}")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        saved_paths.append(str(mp4_path))

    if video_format in ("gif", "both"):
        gif_path = out_dir / f"{file_stem}.gif"
        subsampled = frames[::2]
        if len(frames) > 1 and (len(frames) - 1) % 2 != 0:
            subsampled = subsampled + [frames[-1]]
        pil_frames = [Image.fromarray(f) for f in subsampled]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=100,
            loop=0,
            optimize=True,
        )
        saved_paths.append(str(gif_path))

    return saved_paths


def classify_failure(diag: dict, flags: dict) -> str:
    """根据最终环境诊断与分阶段标志位判定首要失败原因。"""
    if (flags["grasped_wrong_obj"] and not flags["grasped_target"]) or diag["other_in_target_tray"]:
        return "wrong_object"
    if flags["grasped_distractor"] and not flags["grasped_target"]:
        return "distractor_hijack"
    if flags["placed_wrong_tray"]:
        return "wrong_tray"
    if diag["target_in_target_tray"]:
        return "release_or_retreat_fail"
    if flags["grasped_target"] and flags["barrier_collision"]:
        return "barrier_drop"
    if flags["grasped_target"]:
        return "transit_drop"
    if flags["reached_target"]:
        return "grasp_fail"
    return "reach_timeout"


def aggregate_suite_metrics(name: str, ep_records: list[dict]) -> dict:
    """汇总单个套件（或全部回合）的成功率、阶段率、延迟与失败分布。"""
    n = len(ep_records)
    if n == 0:
        return {
            "suite": name,
            "episodes": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "reach_rate": 0.0,
            "grasp_rate": 0.0,
            "cross_rate": 0.0,
            "barrier_collision_rate": 0.0,
            "avg_steps": 0.0,
            "avg_infer_ms": 0.0,
            "p95_infer_ms": 0.0,
            "fps": 0.0,
            "failure_counts": {},
        }

    succ_cnt = sum(1 for r in ep_records if r["success"])
    reach_cnt = sum(1 for r in ep_records if r["reached_target"])
    grasp_cnt = sum(1 for r in ep_records if r["grasped_target"])
    cross_cnt = sum(1 for r in ep_records if r["crossed_barrier"])
    barrier_cnt = sum(1 for r in ep_records if r["barrier_collision"])

    all_latencies = [ms for r in ep_records for ms in r["_step_latencies_ms"]]
    avg_ms = float(np.mean(all_latencies)) if all_latencies else 0.0
    p95_ms = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    fail_counter = Counter(
        r["failure_reason"] for r in ep_records if not r["success"] and r["failure_reason"]
    )
    return {
        "suite": name,
        "episodes": n,
        "success_count": succ_cnt,
        "success_rate": round(succ_cnt / n * 100.0, 2),
        "reach_rate": round(reach_cnt / n * 100.0, 2),
        "grasp_rate": round(grasp_cnt / n * 100.0, 2),
        "cross_rate": round(cross_cnt / n * 100.0, 2),
        "barrier_collision_rate": round(barrier_cnt / n * 100.0, 2),
        "avg_steps": round(float(np.mean([r["steps"] for r in ep_records])), 1),
        "avg_infer_ms": round(avg_ms, 2),
        "p95_infer_ms": round(p95_ms, 2),
        "fps": round(fps, 1),
        "failure_counts": dict(fail_counter),
    }


def build_markdown_report(
    meta: dict, suite_summaries: list[dict], overall_summary: dict
) -> str:
    """生成可直接粘贴进 README 与简历复盘的 Markdown 实验对比表格。"""
    lines = [
        "# Closed-Loop Evaluation Benchmark Report",
        "",
        f"- **Policy Type**: `{meta['policy_type']}`",
        f"- **Checkpoint**: `{meta['ckpt_path']}`",
        f"- **Device / Dtype**: `{meta['device']}` / `{meta['dtype']}`",
        f"- **Task Mode / Seed**: `{meta['task_mode']}` / `{meta['seed']}` (episodes per suite: `{meta['episodes_per_suite']}`)",
        f"- **Overrides**: `n_action_steps={meta['n_action_steps']}`, `temporal_ensemble_coeff={meta['temporal_ensemble_coeff']}`, `num_denoising_steps={meta['num_denoising_steps']}`, `compile={meta['compile']}`",
        "",
        "## 1. Suite Performance Summary",
        "",
        "| Suite | Episodes | Success | Reach | Grasp | Cross | Barrier Hit | Avg Steps | Avg ms | P95 ms | FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [*suite_summaries, overall_summary]:
        lines.append(
            f"| {row['suite']} | {row['episodes']} | "
            f"{row['success_rate']:.1f}% ({row['success_count']}/{row['episodes']}) | "
            f"{row['reach_rate']:.1f}% | {row['grasp_rate']:.1f}% | {row['cross_rate']:.1f}% | "
            f"{row['barrier_collision_rate']:.1f}% | {row['avg_steps']:.1f} | "
            f"{row['avg_infer_ms']:.2f} | {row['p95_infer_ms']:.2f} | {row['fps']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Failure Attribution Breakdown",
            "",
            "| Suite | " + " | ".join(FAILURE_REASONS) + " |",
            "| --- | " + " | ".join(["---:"] * len(FAILURE_REASONS)) + " |",
        ]
    )
    for row in [*suite_summaries, overall_summary]:
        counts = row["failure_counts"]
        vals = " | ".join(str(counts.get(reason, 0)) for reason in FAILURE_REASONS)
        lines.append(f"| {row['suite']} | {vals} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    suites_to_run = parse_suites(args.suites)

    policy_type = args.policy_type
    if policy_type == "auto":
        if not args.ckpt_path:
            raise ValueError("When --policy-type=auto, --ckpt-path must be provided.")
        policy_type = detect_policy_type(args.ckpt_path)
    elif policy_type != "expert" and not args.ckpt_path:
        raise ValueError(f"--ckpt-path is required when --policy-type={policy_type}.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.startswith("cuda")
    use_autocast = is_cuda and args.dtype in ("float16", "bfloat16")
    autocast_dtype = getattr(torch, args.dtype) if use_autocast else None

    print(
        f"[Benchmark] policy_type={policy_type} | ckpt={args.ckpt_path or 'privileged_oracle'} | "
        f"device={device} | dtype={args.dtype} | suites={suites_to_run}",
        flush=True,
    )

    policy = preprocessor = postprocessor = expert = None
    if policy_type == "expert":
        expert = PrivilegedSortingExpert(ExpertConfig(randomize=False))
    else:
        policy, preprocessor, postprocessor = load_policy_and_processors(
            policy_type, args.ckpt_path, device, args
        )

    env = SimpleEnv(
        args.xml_path,
        action_type="joint_angle",
        seed=None,
        task_idx=0,
        headless=not args.gui,
        verbose=False,
    )
    substeps = int(round(1.0 / (env.env.dt * 20)))
    img_transform = torchvision.transforms.ToTensor()
    task_rng = np.random.default_rng(args.seed)

    all_records: list[dict] = []
    suite_summaries: list[dict] = []
    viewer_aborted = False

    try:
        for suite_name in suites_to_run:
            if viewer_aborted:
                break
            suite_cfg = SUITES[suite_name]
            suite_records: list[dict] = []
            videos_saved_in_suite = 0

            for ep_idx in range(args.episodes_per_suite):
                if args.gui and not env.env.is_viewer_alive():
                    viewer_aborted = True
                    break

                ep_seed = args.seed + ep_idx
                task_idx = resolve_task_idx(ep_idx, args.task_mode, task_rng)
                ood_dist = suite_cfg["ood_distractor"]
                ood_pos = suite_cfg["ood_position"]

                if suite_cfg["lang_ood"]:
                    env.reset(
                        seed=ep_seed,
                        task_idx=None,
                        instruction=LANG_OOD_INSTRUCTIONS[task_idx],
                        ood_distractor=ood_dist,
                        ood_position=ood_pos,
                    )
                else:
                    env.reset(
                        seed=ep_seed,
                        task_idx=task_idx,
                        instruction=None,
                        ood_distractor=ood_dist,
                        ood_position=ood_pos,
                    )

                if policy_type == "expert":
                    expert.reset(env, np.random.default_rng(ep_seed))
                else:
                    policy.reset()

                flags = {
                    "reached_target": False,
                    "grasped_target": False,
                    "crossed_barrier": False,
                    "placed_correct_tray": False,
                    "placed_wrong_tray": False,
                    "grasped_wrong_obj": False,
                    "grasped_distractor": False,
                    "barrier_collision": False,
                }
                success = False
                steps_taken = 0
                step_latencies_ms: list[float] = []
                record_video = (
                    args.save_video_dir is not None
                    and videos_saved_in_suite < args.max_videos_per_suite
                )
                video_frames: list[np.ndarray] = []
                diag = env.get_eval_diagnostics()

                for step in range(args.max_steps):
                    if policy_type == "expert":
                        agent_img = wrist_img = None
                        if record_video:
                            agent_img, wrist_img = env.grab_image(include_side=args.gui)
                        t0 = time.perf_counter()
                        action_np, failed, _ = expert.get_action(env)
                        step_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                        if failed:
                            steps_taken = step + 1
                            break
                    else:
                        agent_img = wrist_img = None
                        cached_action = (
                            None
                            if (record_video or args.gui)
                            else pop_cached_action(policy, policy_type)
                        )
                        if cached_action is not None:
                            t0 = time.perf_counter()
                            with torch.inference_mode():
                                action = postprocessor(cached_action)
                            step_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                            action_np = action[0].detach().float().cpu().numpy()
                        else:
                            data, agent_img, wrist_img = build_observation(
                                env, policy_type, policy, img_transform, device, include_side=args.gui
                            )
                            if is_cuda:
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            with torch.inference_mode():
                                inp_batch = preprocessor(data)
                                if use_autocast:
                                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                                        action = policy.select_action(inp_batch)
                                else:
                                    action = policy.select_action(inp_batch)
                                action = postprocessor(action)
                            if is_cuda:
                                torch.cuda.synchronize()
                            step_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                            action_np = action[0].detach().float().cpu().numpy()

                    env.step(action_np)
                    for _ in range(substeps):
                        env.step_env()
                        if env.check_barrier_collision():
                            flags["barrier_collision"] = True

                    diag = env.get_eval_diagnostics()
                    if diag["tcp_to_target_xy"] < 0.055 and diag["tcp_z"] < 0.88:
                        flags["reached_target"] = True
                    if diag["target_z"] > 0.835:
                        flags["grasped_target"] = True
                    if diag["target_y"] > 0.02 and diag["target_z"] > 0.830:
                        flags["crossed_barrier"] = True
                    if diag["target_in_target_tray"]:
                        flags["placed_correct_tray"] = True
                    if diag["target_in_wrong_tray"]:
                        flags["placed_wrong_tray"] = True
                    if (
                        diag["other_z"] > 0.835
                        or diag["other_in_target_tray"]
                        or diag["other_in_wrong_tray"]
                    ):
                        flags["grasped_wrong_obj"] = True
                    if diag["redbull_active"] and diag["redbull_z"] > 0.835:
                        flags["grasped_distractor"] = True

                    steps_taken = step + 1
                    if record_video:
                        status_str = "SUCCESS" if diag["success"] else "RUNNING"
                        video_frames.append(
                            compose_dual_view_frame(
                                agent_img=agent_img,
                                wrist_img=wrist_img,
                                suite_name=suite_name,
                                ep_idx=ep_idx,
                                step=steps_taken,
                                max_steps=args.max_steps,
                                instruction=env.instruction,
                                status_text=status_str,
                                barrier_hit=flags["barrier_collision"],
                            )
                        )

                    if args.gui:
                        if not env.env.is_viewer_alive():
                            viewer_aborted = True
                            break
                        env.render(teleop=False, idx=ep_idx)

                    if diag["success"]:
                        success = True
                        if record_video:
                            end_agent, end_wrist = env.grab_image(include_side=False)
                            video_frames.append(
                                compose_dual_view_frame(
                                    agent_img=end_agent,
                                    wrist_img=end_wrist,
                                    suite_name=suite_name,
                                    ep_idx=ep_idx,
                                    step=steps_taken,
                                    max_steps=args.max_steps,
                                    instruction=env.instruction,
                                    status_text="SUCCESS",
                                    barrier_hit=flags["barrier_collision"],
                                )
                            )
                        break

                if viewer_aborted:
                    break

                failure_reason = "" if success else classify_failure(diag, flags)
                if record_video and video_frames:
                    if not success:
                        end_agent, end_wrist = env.grab_image(include_side=False)
                        video_frames.append(
                            compose_dual_view_frame(
                                agent_img=end_agent,
                                wrist_img=end_wrist,
                                suite_name=suite_name,
                                ep_idx=ep_idx,
                                step=steps_taken,
                                max_steps=args.max_steps,
                                instruction=env.instruction,
                                status_text=f"FAIL:{failure_reason}",
                                barrier_hit=flags["barrier_collision"],
                            )
                        )
                    stem = f"{suite_name}_ep{ep_idx:03d}_{'PASS' if success else failure_reason}"
                    saved_vids = save_episode_video(
                        video_frames, Path(args.save_video_dir), stem, args.video_format
                    )
                    videos_saved_in_suite += 1
                    print(f"  [Video] Saved: {', '.join(saved_vids)}", flush=True)

                ep_avg_ms = float(np.mean(step_latencies_ms)) if step_latencies_ms else 0.0
                ep_p95_ms = float(np.percentile(step_latencies_ms, 95)) if step_latencies_ms else 0.0
                ep_rec = {
                    "suite": suite_name,
                    "ep_idx": ep_idx,
                    "seed": ep_seed,
                    "task_idx": task_idx,
                    "instruction": env.instruction,
                    "ood_distractor": ood_dist,
                    "ood_position": ood_pos,
                    "success": success,
                    "steps": steps_taken,
                    "failure_reason": failure_reason,
                    "avg_infer_ms": round(ep_avg_ms, 2),
                    "p95_infer_ms": round(ep_p95_ms, 2),
                    "_step_latencies_ms": step_latencies_ms,
                    **flags,
                }
                suite_records.append(ep_rec)
                all_records.append(ep_rec)

                status_tag = "PASS" if success else f"FAIL ({failure_reason})"
                print(
                    f"[{suite_name} {ep_idx + 1}/{args.episodes_per_suite}] "
                    f"seed={ep_seed} | task={task_idx} | {status_tag} | "
                    f"steps={steps_taken} | barrier={flags['barrier_collision']} | "
                    f"infer={ep_avg_ms:.2f}ms",
                    flush=True,
                )

            suite_summaries.append(aggregate_suite_metrics(suite_name, suite_records))
    finally:
        env.close()

    overall_summary = aggregate_suite_metrics("ALL", all_records)
    meta = {
        "policy_type": policy_type,
        "ckpt_path": args.ckpt_path or "privileged_oracle",
        "device": device,
        "dtype": args.dtype,
        "task_mode": args.task_mode,
        "seed": args.seed,
        "episodes_per_suite": args.episodes_per_suite,
        "max_steps": args.max_steps,
        "n_action_steps": args.n_action_steps,
        "temporal_ensemble_coeff": args.temporal_ensemble_coeff,
        "num_denoising_steps": args.num_denoising_steps,
        "compile": args.compile,
    }
    report_md = build_markdown_report(meta, suite_summaries, overall_summary)
    print("\n" + report_md)

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 移除临时逐帧延迟列表再序列化 JSON
        serializable_episodes = [
            {k: v for k, v in r.items() if k != "_step_latencies_ms"}
            for r in all_records
        ]
        summary_payload = {
            "meta": meta,
            "suites": suite_summaries,
            "overall": overall_summary,
            "episodes": serializable_episodes,
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

        with open(out_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

        failed_seeds = [
            {
                "seed": r["seed"],
                "task_idx": r["task_idx"],
                "ood_distractor": r["ood_distractor"],
                "ood_position": r["ood_position"],
                "suite": r["suite"],
                "failure_reason": r["failure_reason"],
            }
            for r in all_records
            if not r["success"]
        ]
        with open(out_dir / "failed_seeds.json", "w", encoding="utf-8") as f:
            json.dump(failed_seeds, f, indent=2, ensure_ascii=False)

        print(f"[Output] Saved summary.json, report.md, failed_seeds.json ({len(failed_seeds)} failed) to: {out_dir}")


if __name__ == "__main__":
    main()
