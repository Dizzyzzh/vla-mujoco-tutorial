# Closed-Loop Evaluation Benchmark Report

- **Policy Type**: `expert`
- **Checkpoint**: `privileged_oracle`
- **Device / Dtype**: `cpu` / `auto`
- **Task Mode / Seed**: `cycle` / `100` (episodes per suite: `20`)
- **Overrides**: `n_action_steps=None`, `temporal_ensemble_coeff=None`, `num_denoising_steps=None`, `compile=False`

## 1. Suite Performance Summary

| Suite | Episodes | Success | Reach | Grasp | Cross | Barrier Hit | Avg Steps | Avg ms | P95 ms | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 20 | 100.0% (20/20) | 100.0% | 100.0% | 100.0% | 0.0% | 149.6 | 0.93 | 1.36 | 1073.3 |
| visual_ood | 20 | 100.0% (20/20) | 100.0% | 100.0% | 100.0% | 0.0% | 154.5 | 0.90 | 1.29 | 1108.7 |
| spatial_ood | 20 | 100.0% (20/20) | 100.0% | 100.0% | 100.0% | 0.0% | 151.7 | 0.87 | 1.29 | 1155.5 |
| lang_ood | 20 | 100.0% (20/20) | 100.0% | 100.0% | 100.0% | 0.0% | 149.6 | 0.94 | 1.39 | 1060.8 |
| ALL | 80 | 100.0% (80/80) | 100.0% | 100.0% | 100.0% | 0.0% | 151.3 | 0.91 | 1.34 | 1098.6 |

## 2. Failure Attribution Breakdown

| Suite | wrong_object | distractor_hijack | wrong_tray | release_or_retreat_fail | barrier_drop | transit_drop | grasp_fail | reach_timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| id | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| visual_ood | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| spatial_ood | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| lang_ood | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ALL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
