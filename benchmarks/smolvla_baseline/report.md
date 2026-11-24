# Closed-Loop Evaluation Benchmark Report

- **Policy Type**: `smolvla`
- **Checkpoint**: `./ckpt/smolvla_baseline/best_ema`
- **Device / Dtype**: `cuda` / `bfloat16`
- **Task Mode / Seed**: `cycle` / `100` (episodes per suite: `20`)
- **Overrides**: `n_action_steps=None`, `temporal_ensemble_coeff=None`, `num_denoising_steps=None`, `compile=False`

## 1. Suite Performance Summary

| Suite | Episodes | Success | Reach | Grasp | Cross | Barrier Hit | Avg Steps | Avg ms | P95 ms | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 20 | 5.0% (1/20) | 25.0% | 40.0% | 5.0% | 60.0% | 387.2 | 39.67 | 203.22 | 25.2 |
| visual_ood | 20 | 10.0% (2/20) | 55.0% | 30.0% | 15.0% | 35.0% | 375.1 | 48.78 | 253.97 | 20.5 |
| spatial_ood | 20 | 15.0% (3/20) | 35.0% | 35.0% | 20.0% | 30.0% | 367.9 | 53.86 | 282.60 | 18.6 |
| lang_ood | 20 | 10.0% (2/20) | 40.0% | 20.0% | 20.0% | 20.0% | 377.3 | 43.21 | 235.04 | 23.1 |
| ALL | 80 | 10.0% (8/80) | 38.8% | 31.2% | 15.0% | 36.2% | 376.9 | 46.29 | 255.56 | 21.6 |

## 2. Failure Attribution Breakdown

| Suite | wrong_object | distractor_hijack | wrong_tray | release_or_retreat_fail | barrier_drop | transit_drop | grasp_fail | reach_timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 5 | 0 | 0 | 0 | 6 | 1 | 0 | 7 |
| visual_ood | 5 | 1 | 0 | 0 | 3 | 1 | 4 | 4 |
| spatial_ood | 4 | 0 | 0 | 0 | 3 | 1 | 0 | 9 |
| lang_ood | 5 | 0 | 1 | 0 | 1 | 0 | 4 | 7 |
| ALL | 19 | 1 | 1 | 0 | 13 | 3 | 8 | 27 |
