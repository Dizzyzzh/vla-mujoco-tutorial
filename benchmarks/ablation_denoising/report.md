# Flow Matching Denoising Steps Ablation Report (Success vs. Latency Pareto)

- **Model**: `SmolVLA v3` (`./ckpt/smolvla_v3`, `chunk_size=30, n_action_steps=15`)
- **Suite / Seed**: `id` / `100` (20 episodes per setting)
- **Device / Dtype**: `cuda` (`NVIDIA GeForce RTX 4090`) / `bfloat16`

## 1. Suite Performance & Inference Throughput

| `num_denoising_steps` | Success | Reach | Grasp | Cross | Barrier Hit | Avg Steps | Avg ms | P95 ms | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` (default) | 45.0% (9/20) | **85.0%** | 60.0% | **50.0%** | 30.0% | 293.1 | 13.74 | 192.17 | 72.8 |
| `5` | 20.0% (4/20) | 75.0% | 50.0% | 25.0% | 15.0% | 352.6 | 7.61 | 108.36 | 131.5 |
| **`3` (Pareto Sweet Spot)** | **50.0% (10/20)** | **85.0%** | 60.0% | **50.0%** | **5.0%** | **281.7** | 5.51 | 78.19 | 181.5 |
| `1` | 35.0% (7/20) | 80.0% | **70.0%** | 45.0% | 25.0% | 314.6 | **3.33** | **46.59** | **300.4** |

## 2. Failure Attribution Breakdown

| `num_denoising_steps` | wrong_object | distractor_hijack | wrong_tray | release_or_retreat_fail | barrier_drop | transit_drop | grasp_fail | reach_timeout |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | 2 | 0 | 0 | 0 | 2 | **1** | 6 | **0** |
| `5` | **0** | 0 | 0 | 0 | 3 | 3 | 6 | 4 |
| `3` | 1 | 0 | 0 | 0 | **1** | **1** | 5 | 2 |
| `1` | 1 | 0 | 0 | 0 | 2 | 5 | **2** | 3 |

## 3. Key Findings

1. **3-Step Flow Matching Achieves the Optimal Pareto Frontier**: Reducing Euler integration from `10` to `3` steps cuts average inference latency by **2.49×** (`13.74 ms -> 5.51 ms`, **181.5 FPS**, chunk latency `192 ms -> 78 ms`) while raising ID success to **50.0% (10/20)** and lowering barrier collisions from `30.0%` to **`5.0%`**.
2. **Real-Time 300 FPS Single-Step Generation Remains Viable**: Even with a single Euler step (`steps=1`), `SmolVLA v3` achieves `35.0%` success (`70.0%` grasp) at **`3.33 ms / 300.4 FPS`**, validating the smooth rectified flow field learned on the 30-step action chunk.
