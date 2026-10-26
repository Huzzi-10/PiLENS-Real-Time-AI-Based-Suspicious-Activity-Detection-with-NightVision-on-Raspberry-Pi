# PiLENS KD results (kd v1.0 (2026-10-13), FULL)

Official UCF-Crime split, val-chosen threshold, 5 seeds (mean +- std), test reported once. Lambdas pre-fixed (1.0).

## Binary

| Model | Pi? | Video AUC | Frame AUC | Frame AP | F1 | Normal FPR |
|---|---|---|---|---|---|---|
| teacher_clip_mlp | no (CLIP ViT-B/16) | 0.968 ± 0.002 | 0.828 ± 0.002 | 0.235 ± 0.004 | 0.904 ± 0.006 | 0.068 ± 0.008 |
| teacher_clip_tconv | no (CLIP ViT-B/16) | 0.968 ± 0.005 | 0.831 ± 0.005 | 0.238 ± 0.011 | 0.889 ± 0.018 | 0.055 ± 0.024 |
| student_mlp_base | yes | 0.919 ± 0.004 | 0.824 ± 0.009 | 0.237 ± 0.007 | 0.831 ± 0.007 | 0.164 ± 0.056 |
| student_mlp_kd_score | yes | 0.929 ± 0.001 | 0.787 ± 0.004 | 0.205 ± 0.005 | 0.806 ± 0.015 | 0.077 ± 0.027 |
| student_mlp_kd_feat | yes | 0.921 ± 0.001 | 0.818 ± 0.009 | 0.233 ± 0.008 | 0.825 ± 0.011 | 0.163 ± 0.047 |
| student_mlp_kd_both | yes | 0.929 ± 0.002 | 0.790 ± 0.003 | 0.207 ± 0.005 | 0.807 ± 0.028 | 0.068 ± 0.034 |
| student_tconv_base | yes | 0.916 ± 0.009 | 0.850 ± 0.007 | 0.277 ± 0.006 | 0.813 ± 0.024 | 0.145 ± 0.054 |
| student_tconv_kd_score | yes | 0.921 ± 0.005 | 0.811 ± 0.010 | 0.222 ± 0.013 | 0.790 ± 0.030 | 0.108 ± 0.048 |
| student_tconv_kd_feat | yes | 0.914 ± 0.009 | 0.841 ± 0.008 | 0.266 ± 0.012 | 0.825 ± 0.016 | 0.149 ± 0.035 |
| student_tconv_kd_both | yes | 0.921 ± 0.007 | 0.809 ± 0.014 | 0.218 ± 0.018 | 0.817 ± 0.022 | 0.119 ± 0.046 |

## Paired bootstrap (seed-averaged scores)

| Comparison | Frame AUC diff [95% CI] | Video AUC diff [95% CI] |
|---|---|---|
| student_mlp_kd_score vs student_mlp_base | -0.0417 [-0.0874, -0.0056] | +0.0076 [-0.0092, +0.0235] |
| student_mlp_kd_feat vs student_mlp_base | -0.0041 [-0.0103, +0.0017] | +0.0005 [-0.0066, +0.0075] |
| student_mlp_kd_both vs student_mlp_base | -0.0378 [-0.0804, -0.0036] | +0.0099 [-0.0070, +0.0262] |
| teacher_clip_mlp vs student_mlp_base | -0.0034 [-0.0450, +0.0363] | +0.0435 [+0.0166, +0.0736] |
| student_tconv_kd_score vs student_tconv_base | -0.0386 [-0.0918, +0.0013] | +0.0048 [-0.0116, +0.0210] |
| student_tconv_kd_feat vs student_tconv_base | -0.0073 [-0.0199, +0.0052] | +0.0024 [-0.0070, +0.0124] |
| student_tconv_kd_both vs student_tconv_base | -0.0382 [-0.0893, +0.0014] | +0.0061 [-0.0109, +0.0236] |
| teacher_clip_mlp vs student_tconv_base | -0.0268 [-0.0712, +0.0175] | +0.0458 [+0.0167, +0.0769] |

## classes (accuracy %, macro-F1 %, top-3 %)

| Model | Acc | Macro-F1 | Top-3 |
|---|---|---|---|
| teacher_clip_mil_topk | 44.0 ± 1.5 | 38.0 ± 1.4 | 67.9 ± 1.5 |
| teacher_clip_mil_topk_wrs | 48.7 ± 3.7 | 39.1 ± 3.8 | 69.7 ± 1.8 |
| teacher_clip_linear | 39.7 ± 2.0 | 32.0 ± 3.0 | 65.3 ± 1.8 |
| teacher_clip_linear_wrs | 42.9 ± 1.8 | 33.9 ± 2.0 | 66.6 ± 1.4 |
| student_base_mil_topk | 35.0 ± 2.5 | 26.0 ± 2.5 | 60.6 ± 1.0 |
| student_base_mil_topk_wrs | 35.7 ± 4.5 | 27.8 ± 3.4 | 60.6 ± 2.2 |
| student_base_linear | 37.6 ± 1.7 | 30.5 ± 1.2 | 62.1 ± 0.9 |
| student_base_linear_wrs | 34.7 ± 2.9 | 28.1 ± 1.2 | 59.9 ± 0.5 |
| student_kd_mil_topk | 34.7 ± 1.2 | 25.7 ± 3.1 | 62.6 ± 2.0 |
| student_kd_mil_topk_wrs | 36.6 ± 2.1 | 28.8 ± 1.0 | 63.7 ± 1.2 |
| student_kd_linear | 35.4 ± 1.7 | 27.8 ± 0.8 | 63.7 ± 1.7 |
| student_kd_linear_wrs | 32.3 ± 0.8 | 27.2 ± 0.5 | 62.3 ± 2.9 |

## groups (accuracy %, macro-F1 %, top-3 %)

| Model | Acc | Macro-F1 | Top-3 |
|---|---|---|---|
| teacher_clip_mil_topk | 66.0 ± 2.1 | 65.9 ± 2.5 | 95.6 ± 0.7 |
| teacher_clip_mil_topk_wrs | 67.7 ± 0.8 | 67.6 ± 1.0 | 94.9 ± 0.8 |
| teacher_clip_linear | 61.1 ± 1.1 | 59.7 ± 1.1 | 92.6 ± 1.0 |
| teacher_clip_linear_wrs | 60.9 ± 1.3 | 59.7 ± 1.3 | 92.7 ± 1.8 |
| student_base_mil_topk | 63.3 ± 1.9 | 63.0 ± 2.0 | 94.3 ± 1.4 |
| student_base_mil_topk_wrs | 60.7 ± 2.9 | 60.5 ± 2.9 | 94.1 ± 1.8 |
| student_base_linear | 59.1 ± 2.0 | 58.5 ± 2.1 | 92.1 ± 0.9 |
| student_base_linear_wrs | 55.1 ± 3.5 | 54.5 ± 3.5 | 92.6 ± 0.6 |
| student_kd_mil_topk | 63.3 ± 2.1 | 62.8 ± 2.2 | 94.4 ± 0.9 |
| student_kd_mil_topk_wrs | 59.1 ± 2.8 | 58.5 ± 2.8 | 94.0 ± 2.0 |
| student_kd_linear | 57.1 ± 1.3 | 55.6 ± 1.4 | 94.3 ± 0.9 |
| student_kd_linear_wrs | 58.9 ± 0.7 | 57.1 ± 1.1 | 95.0 ± 0.9 |
