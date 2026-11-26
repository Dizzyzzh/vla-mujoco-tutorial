"""
SmolVLA 交互式开窗部署脚本（支持自定义英文指令与 OOD 干扰测试）。
"""

import argparse
import time

import numpy as np
import torch
import torchvision

from mujoco_env.y_env import SimpleEnv
from scripts.eval_benchmark import build_observation, load_smolvla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLA 交互式 GUI 闭环演示与自定义指令测试")
    parser.add_argument("--ckpt-path", type=str, default="./ckpt/smolvla_v3", help="SmolVLA checkpoint 目录")
    parser.add_argument("--num-episodes", type=int, default=5, help="连续演示回合数")
    parser.add_argument("--max-steps", type=int, default=400, help="单回合最大步数（20Hz 下 400 步=20s）")
    parser.add_argument("--seed", type=int, default=None, help="起始随机种子（默认每回合随机）")
    parser.add_argument("--task-idx", type=int, choices=[0, 1, 2, 3], default=None, help="固定测试任务编号（0~3）")
    parser.add_argument("--instruction", type=str, default=None, help="可选：自定义英文指令")
    parser.add_argument("--ood-distractor", action="store_true", help="加入红牛罐干扰物")
    parser.add_argument("--ood-position", action="store_true", help="扩大物体随机位置边界")
    parser.add_argument("--n-action-steps", type=int, default=None, help="可选：覆盖重规划周期步数")
    parser.add_argument("--num-denoising-steps", type=int, default=None, help="可选：覆盖 Flow Matching 去噪步数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, preprocessor, postprocessor = load_smolvla(
        args.ckpt_path, device, args.n_action_steps, args.num_denoising_steps
    )

    env = SimpleEnv(
        "./asset/example_scene_y.xml",
        action_type="joint_angle",
        seed=args.seed,
        task_idx=args.task_idx,
        ood_distractor=args.ood_distractor,
        ood_position=args.ood_position,
        headless=False,
        verbose=True,
    )
    substeps = int(round(1.0 / (env.env.dt * 20)))
    img_transform = torchvision.transforms.ToTensor()
    results: list[bool] = []
    infer_ms: list[float] = []

    try:
        for ep in range(args.num_episodes):
            if not env.env.is_viewer_alive():
                break

            ep_seed = None if args.seed is None else args.seed + ep
            env.reset(
                seed=ep_seed,
                task_idx=None if args.instruction else args.task_idx,
                instruction=args.instruction,
                ood_distractor=args.ood_distractor,
                ood_position=args.ood_position,
            )
            policy.reset()

            success = False
            barrier_hit = False
            steps_taken = 0
            for step in range(args.max_steps):
                if not env.env.is_viewer_alive():
                    break

                data = build_observation(env, policy, img_transform, device, include_side=True)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    action = postprocessor(policy.select_action(preprocessor(data)))
                infer_ms.append((time.perf_counter() - t0) * 1000.0)

                env.step(action[0].detach().float().cpu().numpy())
                for _ in range(substeps):
                    env.step_env()
                    if env.check_barrier_collision():
                        barrier_hit = True

                steps_taken = step + 1
                env.render(teleop=False, idx=ep)
                if env.check_success():
                    success = True
                    break

            results.append(success)
            print(
                f"[Episode {ep + 1}/{args.num_episodes}] task='{env.instruction}' | "
                f"success={success} | steps={steps_taken} | barrier_hit={barrier_hit}"
            )
    finally:
        env.close()

    if results:
        avg_ms = float(np.mean(infer_ms)) if infer_ms else 0.0
        print(
            f"\n[Summary] Success: {sum(results)}/{len(results)} ({sum(results) / len(results) * 100:.1f}%) | "
            f"Avg infer: {avg_ms:.2f} ms"
        )


if __name__ == "__main__":
    main()
