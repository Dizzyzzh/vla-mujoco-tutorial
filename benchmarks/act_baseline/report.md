# Closed-Loop Evaluation Benchmark Report

- **Policy Type**: `act`
- **Checkpoint**: `./ckpt/act_baseline/best_ema`
- **Device / Dtype**: `cuda` / `auto`
- **Task Mode / Seed**: `cycle` / `100` (episodes per suite: `20`)
- **Overrides**: `n_action_steps=None`, `temporal_ensemble_coeff=None`, `num_denoising_steps=None`, `compile=False`

## 1. Suite Performance Summary

| Suite | Episodes | Success | Reach | Grasp | Cross | Barrier Hit | Avg Steps | Avg ms | P95 ms | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 20 | 0.0% (0/20) | 65.0% | 70.0% | 30.0% | 85.0% | 400.0 | 2.50 | 9.54 | 399.7 |
| visual_ood | 20 | 0.0% (0/20) | 25.0% | 15.0% | 5.0% | 90.0% | 400.0 | 1.09 | 9.36 | 916.6 |
| spatial_ood | 20 | 0.0% (0/20) | 35.0% | 30.0% | 5.0% | 75.0% | 400.0 | 1.05 | 9.18 | 949.8 |
| lang_ood | 20 | 0.0% (0/20) | 60.0% | 55.0% | 25.0% | 70.0% | 400.0 | 1.13 | 9.32 | 884.5 |
| ALL | 80 | 0.0% (0/80) | 46.2% | 42.5% | 16.2% | 80.0% | 400.0 | 1.44 | 9.33 | 692.5 |

## 2. Failure Attribution Breakdown

| Suite | wrong_object | distractor_hijack | wrong_tray | release_or_retreat_fail | barrier_drop | transit_drop | grasp_fail | reach_timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 7 | 0 | 0 | 1 | 11 | 1 | 0 | 0 |
| visual_ood | 4 | 2 | 0 | 0 | 3 | 0 | 2 | 9 |
| spatial_ood | 11 | 0 | 0 | 0 | 5 | 0 | 1 | 3 |
| lang_ood | 6 | 0 | 0 | 0 | 7 | 4 | 0 | 3 |
| ALL | 28 | 2 | 0 | 1 | 26 | 5 | 3 | 15 |
