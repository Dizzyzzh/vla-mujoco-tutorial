# Closed-Loop Evaluation Benchmark Report

- **Policy Type**: `smolvla`
- **Checkpoint**: `./ckpt/smolvla_v2`
- **Device / Dtype**: `cuda` / `bfloat16`
- **Task Mode / Seed**: `cycle` / `100` (episodes per suite: `20`)
- **Overrides**: `n_action_steps=None`, `temporal_ensemble_coeff=None`, `num_denoising_steps=None`, `compile=False`

## 1. Suite Performance Summary

| Suite | Episodes | Success | Reach | Grasp | Cross | Barrier Hit | Avg Steps | Avg ms | P95 ms | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 20 | 30.0% (6/20) | 60.0% | 60.0% | 40.0% | 15.0% | 329.2 | 14.29 | 190.21 | 70.0 |
| visual_ood | 20 | 5.0% (1/20) | 60.0% | 30.0% | 15.0% | 20.0% | 388.9 | 14.27 | 195.88 | 70.1 |
| spatial_ood | 20 | 20.0% (4/20) | 65.0% | 45.0% | 20.0% | 20.0% | 351.8 | 15.70 | 207.48 | 63.7 |
| lang_ood | 20 | 0.0% (0/20) | 45.0% | 20.0% | 5.0% | 0.0% | 400.0 | 17.97 | 216.64 | 55.6 |
| ALL | 80 | 13.8% (11/80) | 57.5% | 38.8% | 20.0% | 13.8% | 367.5 | 15.62 | 200.55 | 64.0 |

## 2. Failure Attribution Breakdown

| Suite | wrong_object | distractor_hijack | wrong_tray | release_or_retreat_fail | barrier_drop | transit_drop | grasp_fail | reach_timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 4 | 0 | 1 | 0 | 1 | 2 | 2 | 4 |
| visual_ood | 4 | 0 | 0 | 0 | 0 | 5 | 5 | 5 |
| spatial_ood | 1 | 0 | 0 | 0 | 3 | 2 | 4 | 6 |
| lang_ood | 4 | 0 | 1 | 0 | 0 | 3 | 5 | 7 |
| ALL | 13 | 0 | 2 | 0 | 4 | 12 | 16 | 22 |
