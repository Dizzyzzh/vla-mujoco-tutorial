# Closed-Loop Evaluation Benchmark Report

- **Policy Type**: `smolvla`
- **Checkpoint**: `./ckpt/smolvla_v3`
- **Device / Dtype**: `cuda` / `bfloat16`
- **Task Mode / Seed**: `cycle` / `100` (episodes per suite: `20`)
- **Overrides**: `n_action_steps=None`, `temporal_ensemble_coeff=None`, `num_denoising_steps=None`, `compile=False`

## 1. Suite Performance Summary

| Suite | Episodes | Success | Reach | Grasp | Cross | Barrier Hit | Avg Steps | Avg ms | P95 ms | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 20 | 45.0% (9/20) | 85.0% | 60.0% | 50.0% | 30.0% | 293.1 | 13.74 | 192.17 | 72.8 |
| visual_ood | 20 | 40.0% (8/20) | 75.0% | 70.0% | 40.0% | 20.0% | 306.0 | 13.53 | 189.67 | 73.9 |
| spatial_ood | 20 | 20.0% (4/20) | 85.0% | 70.0% | 25.0% | 15.0% | 352.6 | 13.56 | 190.36 | 73.8 |
| lang_ood | 20 | 0.0% (0/20) | 55.0% | 35.0% | 15.0% | 10.0% | 400.0 | 13.25 | 187.57 | 75.5 |
| ALL | 80 | 26.2% (21/80) | 75.0% | 58.8% | 32.5% | 18.8% | 337.9 | 13.50 | 188.65 | 74.1 |

## 2. Failure Attribution Breakdown

| Suite | wrong_object | distractor_hijack | wrong_tray | release_or_retreat_fail | barrier_drop | transit_drop | grasp_fail | reach_timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 2 | 0 | 0 | 0 | 2 | 1 | 6 | 0 |
| visual_ood | 3 | 0 | 0 | 0 | 3 | 3 | 2 | 1 |
| spatial_ood | 1 | 0 | 0 | 0 | 3 | 7 | 3 | 2 |
| lang_ood | 5 | 0 | 2 | 0 | 1 | 4 | 4 | 4 |
| ALL | 11 | 0 | 2 | 0 | 9 | 15 | 15 | 7 |
