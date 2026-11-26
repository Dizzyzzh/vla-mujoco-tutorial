"""
SmolVLA 无界面闭环评测基准（4 维度 OOD 套件 + 8 类失败归因 + 推理消融）。

评测套件：
  - id          : 标准训练分布
  - visual_ood  : 额外放置红牛罐干扰物
  - spatial_ood : 扩大桌面物体边界采样范围
  - lang_ood    : 未见过的同义改写英文指令

单步闭环推理流程：
  build_observation -> preprocessor -> policy.select_action -> postprocessor -> env.step
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from mujoco_env.y_env import SimpleEnv

LANG_OOD_INSTRUCTIONS = [
    "Grab the red fruit and drop it into the far container.",
    "Move the red fruit over the wall into the near tray.",
    "Take the citrus fruit and put it in the far container.",
    "Grab the citrus fruit and place it into the near tray.",
]

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
    parser = argparse.ArgumentParser(description="SmolVLA 四维闭环基准评测与失败归因")
    parser.add_argument("--ckpt-path", type=str, required=True, help="SmolVLA checkpoint 目录")
    parser.add_argument("--suites", type=str, default="all", help="评测套件：all 或逗号分隔子集")
    parser.add_argument("--episodes-per-suite", type=int, default=20, help="每个套件运行回合数")
    parser.add_argument("--max-steps", type=int, default=400, help="单回合最大步数（20Hz 下 400 步=20s）")
    parser.add_argument("--seed", type=int, default=100, help="起始随机种子")
    parser.add_argument("--n-action-steps", type=int, default=None, help="可选：覆盖重规划周期步数")
    parser.add_argument("--num-denoising-steps", type=int, default=None, help="可选：覆盖 Flow Matching 去噪步数")
    parser.add_argument("--output-dir", type=str, default=None, help="可选：保存 summary.json 的目录")
    return parser.parse_args()


def load_smolvla(
    ckpt_path: str,
    device: str,
    n_action_steps: int | None = None,
    num_denoising_steps: int | None = None,
):
    """加载 SmolVLA 权重及配套预/后处理器，支持覆盖重规划步数与去噪步数。"""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = SmolVLAConfig.from_pretrained(ckpt_path)
    cfg.device = device
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    if num_denoising_steps is not None:
        cfg.num_steps = num_denoising_steps

    policy = SmolVLAPolicy.from_pretrained(ckpt_path, config=cfg)
    policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def build_observation(
    env: SimpleEnv,
    policy,
    img_transform,
    device: str,
    include_side: bool = False,
) -> dict:
    """读取当前 7 维关节状态、双目 256x256 图像与英文指令，组装模型输入字典。"""
    state = env.get_joint_state()
    agent_img, wrist_img = env.grab_image(include_side=include_side)
    agent_tensor = img_transform(Image.fromarray(agent_img).resize((256, 256)))

    data = {
        "observation.state": torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0),
        "observation.image": agent_tensor.unsqueeze(0).to(device),
        "task": [env.instruction],
    }
    if "observation.wrist_image" in policy.config.input_features:
        wrist_tensor = img_transform(Image.fromarray(wrist_img).resize((256, 256)))
        data["observation.wrist_image"] = wrist_tensor.unsqueeze(0).to(device)
    return data


def classify_failure(diag: dict, flags: dict) -> str:
    """根据环境诊断与分阶段标志位判定 8 类首要失败原因。"""
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
    """汇总单个套件的成功率、阶段转化率、推理延迟与失败分布。"""
    n = len(ep_records)
    if n == 0:
        return {"suite": name, "episodes": 0, "success_rate": 0.0, "failure_counts": {}}

    all_latencies = [ms for r in ep_records for ms in r["_step_latencies_ms"]]
    avg_ms = float(np.mean(all_latencies)) if all_latencies else 0.0
    p95_ms = float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    succ_cnt = sum(1 for r in ep_records if r["success"])

    return {
        "suite": name,
        "episodes": n,
        "success_count": succ_cnt,
        "success_rate": round(succ_cnt / n * 100.0, 2),
        "reach_rate": round(sum(1 for r in ep_records if r["reached_target"]) / n * 100.0, 2),
        "grasp_rate": round(sum(1 for r in ep_records if r["grasped_target"]) / n * 100.0, 2),
        "cross_rate": round(sum(1 for r in ep_records if r["crossed_barrier"]) / n * 100.0, 2),
        "barrier_collision_rate": round(sum(1 for r in ep_records if r["barrier_collision"]) / n * 100.0, 2),
        "avg_steps": round(float(np.mean([r["steps"] for r in ep_records])), 1),
        "avg_infer_ms": round(avg_ms, 2),
        "p95_infer_ms": round(p95_ms, 2),
        "fps": round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
        "failure_counts": dict(Counter(r["failure_reason"] for r in ep_records if not r["success"])),
    }


def run_episode(
    env: SimpleEnv,
    policy,
    preprocessor,
    postprocessor,
    img_transform,
    device: str,
    max_steps: int,
    substeps: int,
) -> dict:
    """执行单回合闭环控制，追踪阶段状态、单步推理耗时与失败归因。"""
    policy.reset()
    flags = {
        "reached_target": False,
        "grasped_target": False,
        "crossed_barrier": False,
        "placed_wrong_tray": False,
        "grasped_wrong_obj": False,
        "grasped_distractor": False,
        "barrier_collision": False,
    }
    latencies_ms: list[float] = []
    success = False
    steps_taken = 0
    diag = env.get_eval_diagnostics()

    for step in range(max_steps):
        data = build_observation(env, policy, img_transform, device, include_side=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            action = postprocessor(policy.select_action(preprocessor(data)))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        env.step(action[0].detach().float().cpu().numpy())
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
        if diag["target_in_wrong_tray"]:
            flags["placed_wrong_tray"] = True
        if diag["other_z"] > 0.835 or diag["other_in_target_tray"] or diag["other_in_wrong_tray"]:
            flags["grasped_wrong_obj"] = True
        if diag["redbull_active"] and diag["redbull_z"] > 0.835:
            flags["grasped_distractor"] = True

        steps_taken = step + 1
        if diag["success"]:
            success = True
            break

    return {
        "success": success,
        "steps": steps_taken,
        "failure_reason": "" if success else classify_failure(diag, flags),
        "_step_latencies_ms": latencies_ms,
        **flags,
    }


def main() -> None:
    args = parse_args()
    suites_to_run = (
        list(SUITES.keys())
        if args.suites.strip().lower() == "all"
        else [s.strip() for s in args.suites.split(",") if s.strip()]
    )
    for name in suites_to_run:
        if name not in SUITES:
            raise ValueError(f"Unknown suite '{name}'. Valid: {list(SUITES.keys())}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, preprocessor, postprocessor = load_smolvla(
        args.ckpt_path, device, args.n_action_steps, args.num_denoising_steps
    )
    env = SimpleEnv(
        "./asset/example_scene_y.xml", action_type="joint_angle", headless=True, verbose=False
    )
    substeps = int(round(1.0 / (env.env.dt * 20)))
    img_transform = torchvision.transforms.ToTensor()

    all_records: list[dict] = []
    suite_summaries: list[dict] = []
    try:
        for suite_name in suites_to_run:
            cfg = SUITES[suite_name]
            suite_records: list[dict] = []
            for ep_idx in range(args.episodes_per_suite):
                ep_seed = args.seed + ep_idx
                task_idx = ep_idx % len(SimpleEnv.TASKS)
                env.reset(
                    seed=ep_seed,
                    task_idx=None if cfg["lang_ood"] else task_idx,
                    instruction=LANG_OOD_INSTRUCTIONS[task_idx] if cfg["lang_ood"] else None,
                    ood_distractor=cfg["ood_distractor"],
                    ood_position=cfg["ood_position"],
                )
                rec = run_episode(
                    env, policy, preprocessor, postprocessor, img_transform, device, args.max_steps, substeps
                )
                rec.update({"suite": suite_name, "ep_idx": ep_idx, "seed": ep_seed, "task_idx": task_idx})
                suite_records.append(rec)
                all_records.append(rec)
                tag = "PASS" if rec["success"] else f"FAIL ({rec['failure_reason']})"
                print(
                    f"[{suite_name} {ep_idx + 1}/{args.episodes_per_suite}] "
                    f"seed={ep_seed} | task={task_idx} | {tag} | steps={rec['steps']}"
                )
            suite_summaries.append(aggregate_suite_metrics(suite_name, suite_records))
    finally:
        env.close()

    overall = aggregate_suite_metrics("ALL", all_records)
    print("\n=== Evaluation Summary ===")
    print(f"{'Suite':<12} {'Success':>14} {'Reach':>7} {'Grasp':>7} {'Cross':>7} {'BarrierHit':>11} {'AvgMs':>7} {'FPS':>6}")
    for row in [*suite_summaries, overall]:
        succ_str = f"{row['success_rate']:.1f}% ({row['success_count']}/{row['episodes']})"
        print(
            f"{row['suite']:<12} {succ_str:>14} {row['reach_rate']:>6.1f}% {row['grasp_rate']:>6.1f}% "
            f"{row['cross_rate']:>6.1f}% {row['barrier_collision_rate']:>10.1f}% "
            f"{row['avg_infer_ms']:>7.2f} {row['fps']:>6.1f}"
        )
    print("Failure Counts (ALL):", overall["failure_counts"])

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ckpt_path": args.ckpt_path,
            "suites": suite_summaries,
            "overall": overall,
            "episodes": [{k: v for k, v in r.items() if k != "_step_latencies_ms"} for r in all_records],
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[Output] Saved summary.json to: {out_dir}")


if __name__ == "__main__":
    main()
