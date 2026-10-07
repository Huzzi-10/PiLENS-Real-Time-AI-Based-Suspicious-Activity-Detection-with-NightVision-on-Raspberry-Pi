# Paper tables (fill only with measured numbers)

Har cell ke saath likha hai ke number kis file / field se aata hai. Jo cell measure nahi hua wo
khali rahe ya "est." ke saath alag likha jaye (locked decision 9).

## Table 1: 14-class recognition (Task A)

Official 4 folds, 38 train / 12 test per class, frozen X3D-S features, mean +- std over folds.
Source: `runs/cls14_<head>/summary.json` -> `accuracy_mean_std`, `macro_f1_mean_std`.

| Model | Head | Accuracy (%) | Macro-F1 (%) |
|---|---|---|---|
| v1 ResNet18-LSTM (baseline) | - | | |
| X3D-S | linear | | |
| X3D-S | mil_topk | | |
| X3D-S | tconv_mil | | |
| MoViNet-A0 | linear | | |
| Chance | | 7.1 | |

Per-class P/R/F1: `per_class`; confusion matrix: `confusion_matrix_first_seed_summed_over_folds`.

## Table 2: weakly supervised anomaly detection (Task B)

Official split 1610 / 290, val = 15% of train, 5 seeds. Source: `runs/mil_<head>/summary.json`.

| Model | Head | Video AUC | Frame AUC | Frame AP | P / R at val threshold |
|---|---|---|---|---|---|
| X3D-S | MLP | | | | `runs[*].test_video_pr_at_val_thr` |
| X3D-S | TConv | | | | |
| X3D-S + Gaussian smoothing (sigma = 1, fixed) | MLP | - | `frame_auc_smoothed` | `frame_ap_smoothed` | |

## Table 3: Raspberry Pi 5 latency (measured)

Source: `bench_results/<stamp>_<tag>.json` -> `latency.*` (median / p95 in ms), `health.temp_c`, `system`.

| Precision | Threads | Cooling | Preprocess | Backbone | Binary head | Stage 1 total (median / p95) | 14-class head | Temp start / max (C) |
|---|---|---|---|---|---|---|---|---|
| FP32 | 4 | fan | | | | | | |
| FP32 | 4 | none | | | | | | |
| INT8 | 4 | fan | | | | | | |

Versions row: `system.os`, `system.python`, `system.onnxruntime`, `system.board`.

## Table 4: full system on the Pi (capture + motion gate + alert thread running)

Source: `logs/<run>/summary.json`.

| Setting | Stream throughput (`capture_fps`) | Clip latency median / p95 (`clip_latency`) | Gate skip (`motion_gate_skip_frac`) | CPU % / RAM MB (`health`) | Temp max |
|---|---|---|---|---|---|
| Cascade + motion gate | | | | | |
| Cascade, no motion gate | | | - | | |
| Binary only (`--no-stage2`) | | | | | |

## Table 5: alert logic (test videos, threshold from validation)

Source: `replay/alert_eval.json` (`python -m pilens_v2.eval.alert_eval`).

| Rule | Motion gate | Detection rate | Event-to-alert median / p95 (s) | False alarms / hour |
|---|---|---|---|---|
| 1 of 1 | on | | | |
| 2 of 3 | on | | | |
| 3 of 5 | on | | | |
| 2 of 3 | off | | | |

## Table 6: ablations

| Ablation | Variant | Video AUC | Frame AUC | 14-class Acc | Note |
|---|---|---|---|---|---|
| Preprocessing | RGB | | | | |
| Preprocessing | gray-3ch (`--gray3`) | | | | |
| Crop | center 160 (default) | | | | |
| Crop | full frame squash (`--crop-mode squash`) | | | | |
| Crop | union box / motion ROI | | | | Track B |
| Test subset | Day only | | | | `splits/day_night.csv` |
| Test subset | Night only | | | | `splits/day_night.csv` |
| Quantization | FP32 vs INT8 | | | | accuracy drop |
