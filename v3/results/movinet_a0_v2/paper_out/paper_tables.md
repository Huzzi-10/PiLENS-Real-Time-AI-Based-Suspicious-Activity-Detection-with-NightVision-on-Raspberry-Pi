
## MoViNet-A0: test 290 videos (anomaly 140), val 242

### Table 1. Binary detection (mean ± std over seeds; thr = val Youden)
| backbone | head | video AUC | frame AUC | frame AP | precision | recall | F1 | n seeds |
|---|---|---|---|---|---|---|---|---|
| MoViNet-A0 | mlp | 0.919 ± 0.003 | 0.828 ± 0.004 | 0.241 ± 0.004 | 0.854 ± 0.026 | 0.781 ± 0.047 | 0.815 ± 0.013 | 5 |
| MoViNet-A0 | tconv | 0.912 ± 0.006 | 0.852 ± 0.006 | 0.282 ± 0.012 | 0.828 ± 0.019 | 0.807 ± 0.021 | 0.817 ± 0.009 | 5 |

### Table 2. Video AUC, mlp seed 0, bootstrap 95% CI (test videos par, B=2000)
- MoViNet-A0: 0.921  [0.889, 0.949]

### Table 3. Operating points (mlp seed 0; thr val se, test par sirf lagaya)
| backbone | thr rule | thr | video TPR | video FPR (normal test) | clip FPR (normal segments) |
|---|---|---|---|---|---|
| MoViNet-A0 | Youden (deployed) | 0.994 | 0.757 | 0.113 | 0.0092 |
| MoViNet-A0 | val FPR=5% | 1.000 | 0.629 | 0.033 | 0.0021 |
| MoViNet-A0 | val FPR=10% | 0.999 | 0.736 | 0.093 | 0.0063 |

### Table 4. Alert rule (n lagataar segments mein se k, thr = Youden, test videos par)
| backbone | rule | video TPR | video FPR |
|---|---|---|---|
| MoViNet-A0 | 1 of 1 | 0.757 | 0.113 |
| MoViNet-A0 | 2 of 2 | 0.621 | 0.027 |
| MoViNet-A0 | 2 of 3 | 0.643 | 0.040 |
| MoViNet-A0 | 3 of 5 | 0.607 | 0.013 |
(note: UCF test mein segments poori video par phaile hain, Pi ke 0.6 s hop se alag; ye rule ka relative asar dikhata hai, per-hour rate nahi.)

### Table 5. Per-category detection (mlp seed 0, Youden thr): recall aur AUC vs normal

MoViNet-A0:
| category | n | recall | AUC vs normal |
|---|---|---|---|
| Abuse | 2 | 1.00 | 0.930 |
| Arrest | 5 | 1.00 | 0.991 |
| Arson | 9 | 0.89 | 0.964 |
| Assault | 3 | 1.00 | 0.976 |
| Burglary | 13 | 0.92 | 0.968 |
| Explosion | 21 | 0.81 | 0.934 |
| Fighting | 5 | 0.80 | 0.975 |
| RoadAccidents | 23 | 0.70 | 0.905 |
| Robbery | 5 | 0.80 | 0.956 |
| Shooting | 23 | 0.87 | 0.950 |
| Shoplifting | 21 | 0.38 | 0.784 |
| Stealing | 5 | 0.80 | 0.957 |
| Vandalism | 5 | 0.60 | 0.939 |

### Table 7. Ablation, MoViNet-A0 (mlp-based, seeds 0-2)
| variant | video AUC | frame AUC | frame AP | F1 |
|---|---|---|---|---|
| full | 0.920 ± 0.001 | 0.827 ± 0.002 | 0.242 ± 0.005 | 0.821 ± 0.012 |
| rank_only | 0.920 ± 0.003 | 0.822 ± 0.009 | 0.239 ± 0.015 | 0.824 ± 0.010 |
| topk1 | 0.916 ± 0.006 | 0.822 ± 0.002 | 0.241 ± 0.008 | 0.823 ± 0.008 |
| topk5 | 0.922 ± 0.002 | 0.820 ± 0.002 | 0.237 ± 0.004 | 0.832 ± 0.017 |
| linear_head | 0.896 ± 0.000 | 0.780 ± 0.002 | 0.195 ± 0.001 | 0.794 ± 0.038 |

### Table 8. Stage 2 (MoViNet-A0)
- official 14-class folds, linear: acc 34.1 ± 3.3 (chance 7.1)
- official 14-class folds, mil_topk: acc 36.4 ± 3.0 (chance 7.1)
- DET 13-class (140 anomaly test videos), linear: acc 36.7 ± 1.7 | top-3 62.1 (chance 7.7)
- DET 13-class (140 anomaly test videos), linear_wrs: acc 34.8 ± 3.7 | top-3 60.0 (chance 7.7)
- DET 13-class (140 anomaly test videos), mil_topk: acc 33.3 ± 1.8 | top-3 61.2 (chance 7.7)
- DET 13-class (140 anomaly test videos), mil_topk_wrs: acc 33.3 ± 4.5 | top-3 60.2 (chance 7.7)

Cascade end-to-end (MoViNet-A0; binary mlp seed 0 -> linear stage 2): anomaly videos detected 0.757 | class correct given detected 0.321 | detected AND correct 0.243 | normal test videos falsely alerted 0.113
(figure skip: No module named 'matplotlib' )