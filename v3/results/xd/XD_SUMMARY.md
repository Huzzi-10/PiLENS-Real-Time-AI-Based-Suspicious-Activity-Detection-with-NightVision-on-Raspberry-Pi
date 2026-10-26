# XD-Violence results (xd v1.2 (2026-10-14))

Dataset: Kaggle XD-Violence Single-Label Edition | train 3710 | test 753 (anomaly 453) | annotation match 453/453

| model | frame AP | frame AUC | video AUC | video AP | recall | normal FPR |
|---|---|---|---|---|---|---|
| a0_mlp | 0.6857 +- 0.0112 | 0.8948 +- 0.0036 | 0.9797 +- 0.0017 | 0.9876 +- 0.0011 | 0.9161 +- 0.0157 | 0.0560 +- 0.0098 |
| a0_tconv | 0.7522 +- 0.0222 | 0.9280 +- 0.0049 | 0.9836 +- 0.0005 | 0.9899 +- 0.0005 | 0.9210 +- 0.0117 | 0.0507 +- 0.0124 |
| clip_mlp | 0.7748 +- 0.0099 | 0.9315 +- 0.0013 | 0.9930 +- 0.0004 | 0.9958 +- 0.0002 | 0.9664 +- 0.0121 | 0.0560 +- 0.0131 |

Bootstrap:
- clip_mlp__vs__a0_mlp: {"video_auc_diff": 0.0127814569536423, "video_ci": [0.005258428012479549, 0.02120252432285726], "frame_ap_diff": 0.08301308245580674, "frame_ap_ci": [0.058000550388334904, 0.10665042720481598]}
- a0_tconv__vs__a0_mlp: {"video_auc_diff": 0.004856512141280356, "video_ci": [-0.00048059612834844826, 0.010722888461595869], "frame_ap_diff": 0.07011568570026405, "frame_ap_ci": [0.05882517312257799, 0.08146423614060168]}

Cascade (mean over seeds, %):
- fpr5_r0: A0 recall 87.8 FPR 3.5 -> A0+CLIP recall 87.1 FPR 2.1
- fpr5_r1: A0 recall 87.8 FPR 3.5 -> A0+CLIP recall 87.0 FPR 2.1
- fpr10_r0: A0 recall 94.0 FPR 7.5 -> A0+CLIP recall 92.8 FPR 3.0
- fpr10_r1: A0 recall 94.0 FPR 7.5 -> A0+CLIP recall 93.0 FPR 3.2
- youden_r0: A0 recall 91.6 FPR 5.6 -> A0+CLIP recall 90.6 FPR 2.6
- youden_r1: A0 recall 91.6 FPR 5.6 -> A0+CLIP recall 90.8 FPR 2.7

Crime type (6 classes, chance 16.7, majority 23.6):
- a0_mil_topk: acc 87.3 | macro F1 78.1
- a0_linear: acc 86.2 | macro F1 78.3
- clip_mil_topk: acc 92.7 | macro F1 85.1
- clip_linear: acc 88.3 | macro F1 81.0
