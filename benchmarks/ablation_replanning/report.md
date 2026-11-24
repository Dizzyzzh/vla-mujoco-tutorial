# Receding-Horizon Replanning Frequency Ablation Report

- **Model**: `SmolVLA v2` (`./ckpt/smolvla_v2`, `chunk_size=30`)
- **Suite / Seed**: `id` / `100` (20 episodes per setting)
- **Device / Dtype**: `cuda` / `bfloat16`

## 1. Closed-Loop Performance & Throughput

| `n_action_steps` | Replan Interval | Success | Reach | Grasp | Cross | Barrier Hit | Avg Steps | Avg ms | P95 ms | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `15` (default) | 0.75 s | **35.0% (7/20)** | **80.0%** | **60.0%** | **40.0%** | **5.0%** | 315.4 | **12.95** | **184.06** | **77.2** |
| `8` | 0.40 s | 20.0% (4/20) | 55.0% | 50.0% | 25.0% | 40.0% | 353.0 | 23.68 | 186.71 | 42.2 |
| `5` | 0.25 s | **35.0% (7/20)** | 65.0% | **60.0%** | 35.0% | 30.0% | **313.4** | 38.98 | 193.63 | 25.7 |

## 2. Failure Attribution Breakdown

| `n_action_steps` | wrong_object | distractor_hijack | wrong_tray | release_or_retreat_fail | barrier_drop | transit_drop | grasp_fail | reach_timeout |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `15` | 3 | 0 | 0 | 0 | **1** | 4 | 4 | **1** |
| `8` | **2** | 0 | 0 | 0 | 3 | **3** | 3 | 5 |
| `5` | 6 | 0 | 0 | 0 | **1** | 4 | **1** | **1** |

## 3. Key Findings

1. **Long-Horizon Chunk Consistency (`n=15`) Minimizes Barrier Collisions**: Executing 15 coherent steps (0.75 s) preserves the smooth quadratic Bézier obstacle-crossing arc, keeping barrier collisions at **5.0%** (`1/20`) and achieving **80.0% Reach** at **77.2 FPS**.
2. **Dense Replanning (`n=5`) Improves Fine Grasp Alignment but Triggers Mode Switching**: Reducing the horizon to 0.25 s cuts `grasp_fail` from `4` down to `1`, but more frequent Flow Matching noise resampling increases mid-trajectory mode switches, raising `wrong_object` from `3` to `6` and barrier collisions from `5.0%` to `30.0%`.
