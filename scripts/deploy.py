import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from mujoco_env.y_env import SimpleEnv

try:
    from scripts.eval_benchmark import (
        build_observation,
        compose_dual_view_frame,
        detect_policy_type,
        load_policy_and_processors,
        save_episode_video,
    )
except ImportError:
    from eval_benchmark import (
        build_observation,
        compose_dual_view_frame,
        detect_policy_type,
        load_policy_and_processors,
        save_episode_video,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单策略（ACT / SmolVLA / PI0.5）交互式闭环部署与自定义语言指令测试脚本"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./ckpt/smolvla_baseline/best_ema",
        help="本地策略权重目录（含 config.json、model.safetensors 与 pre/post processor）",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        choices=["auto", "act", "smolvla", "pi05"],
        default="auto",
        help="策略类型（默认 auto 从 config.json 自动识别）",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default="./asset/example_scene_y.xml",
        help="MuJoCo 跨障分拣场景主 XML 文件路径",
    )
    parser.add_argument("--num-episodes", type=int, default=5, help="连续测试的回合数")
    parser.add_argument("--max-steps", type=int, default=400, help="单回合最大控制步数（20Hz 下 400 步 = 20 秒）")
    parser.add_argument("--seed", type=int, default=None, help="起始随机种子（默认 None 每回合随机初位）")
    parser.add_argument(
        "--task-idx",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="固定测试标准任务编号（0~3）；不传且未指定 --instruction 时随机轮换",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="可选：自由输入英文任务指令（如 'Grab the red fruit and drop it into the far container.'）",
    )
    parser.add_argument("--ood-distractor", action="store_true", help="在抓取区加入红牛罐干扰物")
    parser.add_argument("--ood-position", action="store_true", help="开启扩展空间位置外推采样")

    parser.add_argument("--device", type=str, default=None, help="推理计算设备（默认自动选择 cuda / cpu）")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="推理精度覆盖",
    )
    parser.add_argument("--n-action-steps", type=int, default=None, help="覆盖策略每次连续执行的动作步数")
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="为 ACT 启用时序集成平滑（例如 0.01，自动设 n_action_steps=1）",
    )
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=None,
        help="覆盖 SmolVLA / PI0.5 的 Flow Matching 去噪步数",
    )
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile 加速推理")
    parser.add_argument("--headless", action="store_true", help="无界面离屏运行（适合服务器环境快速部署验证）")
    parser.add_argument("--save-video-dir", type=str, default=None, help="可选：保存部署过程双视角视频/GIF 的目录")
    parser.add_argument(
        "--video-format",
        type=str,
        choices=["mp4", "gif", "both"],
        default="mp4",
        help="视频导出格式（mp4 / gif / both）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_type = detect_policy_type(args.ckpt_path) if args.policy_type == "auto" else args.policy_type
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.startswith("cuda")
    use_autocast = is_cuda and args.dtype in ("float16", "bfloat16")
    autocast_dtype = getattr(torch, args.dtype) if use_autocast else None

    print(f"[Deploy] policy_type={policy_type} | device={device} | ckpt={args.ckpt_path}")
    policy, preprocessor, postprocessor = load_policy_and_processors(
        policy_type, args.ckpt_path, device, args
    )
    print(f"[Deploy] Input features: {list(policy.config.input_features.keys())}")

    env = SimpleEnv(
        args.xml_path,
        action_type="joint_angle",
        seed=args.seed,
        task_idx=args.task_idx,
        ood_distractor=args.ood_distractor,
        ood_position=args.ood_position,
        headless=args.headless,
        verbose=not args.headless,
    )
    substeps = int(round(1.0 / (env.env.dt * 20)))
    img_transform = torchvision.transforms.ToTensor()

    results: list[dict] = []
    infer_times_ms: list[float] = []

    try:
        for ep in range(args.num_episodes):
            if not args.headless and not env.env.is_viewer_alive():
                print("[Info] Viewer closed by user, stopping deployment.")
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

            steps_taken = 0
            success = False
            barrier_hit = False
            video_frames: list[np.ndarray] = []

            for step in range(args.max_steps):
                if not args.headless and not env.env.is_viewer_alive():
                    break

                data, agent_img, wrist_img = build_observation(
                    env, policy_type, policy, img_transform, device, include_side=not args.headless
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
                infer_times_ms.append((time.perf_counter() - t0) * 1000.0)

                action_np = action[0].detach().float().cpu().numpy()
                env.step(action_np)
                for _ in range(substeps):
                    env.step_env()
                    if env.check_barrier_collision():
                        barrier_hit = True

                steps_taken = step + 1
                if not args.headless:
                    env.render(teleop=False, idx=ep)

                done = env.check_success()
                if args.save_video_dir is not None:
                    video_frames.append(
                        compose_dual_view_frame(
                            agent_img=agent_img,
                            wrist_img=wrist_img,
                            suite_name="deploy",
                            ep_idx=ep,
                            step=steps_taken,
                            max_steps=args.max_steps,
                            instruction=env.instruction,
                            status_text="SUCCESS" if done else "RUNNING",
                            barrier_hit=barrier_hit,
                        )
                    )

                if done:
                    success = True
                    break

            if args.save_video_dir is not None and video_frames:
                stem = f"deploy_{policy_type}_ep{ep:03d}_{'PASS' if success else 'FAIL'}"
                saved = save_episode_video(
                    video_frames, Path(args.save_video_dir), stem, args.video_format
                )
                print(f"  [Video] Saved: {', '.join(saved)}")

            results.append({"success": success, "steps": steps_taken, "barrier_hit": barrier_hit})
            print(
                f"[Episode {ep + 1}/{args.num_episodes}] task='{env.instruction}' | "
                f"success={success} | steps={steps_taken} | barrier_hit={barrier_hit}"
            )
    finally:
        env.close()

    if results:
        n = len(results)
        succ_cnt = sum(1 for r in results if r["success"])
        succ_rate = succ_cnt / n * 100.0
        avg_ms = float(np.mean(infer_times_ms)) if infer_times_ms else 0.0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        print("\n========== Deployment Summary ==========")
        print(f"Policy     : {policy_type} ({args.ckpt_path})")
        print(f"Episodes   : {n} (OOD distractor={args.ood_distractor}, OOD pos={args.ood_position})")
        print(f"Success    : {succ_rate:.1f}% ({succ_cnt}/{n})")
        print(f"Inference  : {avg_ms:.2f} ms/step ({fps:.1f} FPS)")
        print("========================================")


if __name__ == "__main__":
    main()
