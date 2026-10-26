# XD-Violence FINAL (xd v1.2, 2026-10-14) - results/xd/xd_out

Kaggle bypktt/xd-violence single-label edition: train 3710 (1 corrupt normal skipped), test 753 (453 anomaly, all 453 matched to official annotations.txt). 5 seeds, thresholds from val only, test once.
NOTE: official XD test = 800 videos (500 anomaly, incl. multi-label); ours is the 753 single-label subset -> literature numbers are NOT strictly comparable (say so in paper).

| model | frame AP (official metric) | frame AUC | video AUC | recall | normal FPR |
|---|---|---|---|---|---|
| A0 mlp (Pi head) | 68.6 +- 1.1 | 89.5 | 0.980 | 91.6 | 5.6 |
| A0 tconv | 75.2 +- 2.2 | 92.8 | 0.984 | 92.1 | 5.1 |
| CLIP mlp | 77.5 +- 1.0 | 93.2 | 0.993 | 96.6 | 5.6 |

Bootstrap: CLIP vs A0 mlp frame AP +8.3 CI [5.8, 10.7] sig; tconv vs mlp +7.0 CI [5.9, 8.1] sig.
Cascade (val thresholds): youden recall 91.6/FPR 5.6 -> 90.6/2.6; fpr10 94.0/7.5 -> 92.8/3.0; fpr5 87.8/3.5 -> 87.1/2.1.
Crime type 6 classes (chance 16.7): A0 mil_topk 87.3 / F1 78.1; CLIP mil_topk 92.7 / F1 85.1.
Literature XD frame AP (from memory, VERIFY): Sultani 73.2, RTFM 77.8, MSL 78.3, MGFN 79.2, S3R 80.3, UR-DMU 81.7, VadCLIP 84.5 (all GPU, I3D/CLIP features, official 800 test).
Old v1.1 output (frame AP NaN) kept in results/xd_v11_nofaframe.
