
## X3D-S: test 290 videos (anomaly 140), val 242

## MoViNet-A0: test 290 videos (anomaly 140), val 242

### Table 1. Binary detection (mean ± std over seeds; thr = val Youden)
| backbone | head | video AUC | frame AUC | frame AP | precision | recall | F1 | n seeds |
|---|---|---|---|---|---|---|---|---|
| X3D-S | mlp | 0.889 ± 0.004 | 0.773 ± 0.008 | 0.183 ± 0.005 | 0.809 ± 0.022 | 0.771 ± 0.036 | 0.789 ± 0.011 | 5 |
| X3D-S | tconv | 0.894 ± 0.007 | 0.790 ± 0.007 | 0.197 ± 0.004 | 0.819 ± 0.027 | 0.803 ± 0.057 | 0.809 ± 0.025 | 5 |
| MoViNet-A0 | mlp | 0.919 ± 0.003 | 0.828 ± 0.004 | 0.241 ± 0.004 | 0.854 ± 0.026 | 0.781 ± 0.047 | 0.815 ± 0.013 | 5 |
| MoViNet-A0 | tconv | 0.912 ± 0.006 | 0.852 ± 0.006 | 0.282 ± 0.012 | 0.828 ± 0.019 | 0.807 ± 0.021 | 0.817 ± 0.009 | 5 |

### Table 2. Video AUC, mlp seed 0, bootstrap 95% CI (test videos par, B=2000)
- X3D-S: 0.884  [0.844, 0.921]
- MoViNet-A0: 0.921  [0.889, 0.949]
- paired difference (MoViNet-A0 - X3D-S): +0.037  [+0.003, +0.072]  (significant at 95%)

### Table 3. Operating points (mlp seed 0; thr val se, test par sirf lagaya)
| backbone | thr rule | thr | video TPR | video FPR (normal test) | clip FPR (normal segments) |
|---|---|---|---|---|---|
| X3D-S | Youden (deployed) | 0.753 | 0.779 | 0.200 | 0.0379 |
| X3D-S | val FPR=5% | 1.000 | 0.550 | 0.087 | 0.0135 |
| X3D-S | val FPR=10% | 0.991 | 0.757 | 0.140 | 0.0273 |
| MoViNet-A0 | Youden (deployed) | 0.994 | 0.757 | 0.113 | 0.0092 |
| MoViNet-A0 | val FPR=5% | 1.000 | 0.629 | 0.033 | 0.0021 |
| MoViNet-A0 | val FPR=10% | 0.999 | 0.736 | 0.093 | 0.0063 |

### Table 4. Alert rule (n lagataar segments mein se k, thr = Youden, test videos par)
| backbone | rule | video TPR | video FPR |
|---|---|---|---|
| X3D-S | 1 of 1 | 0.779 | 0.200 |
| X3D-S | 2 of 2 | 0.700 | 0.127 |
| X3D-S | 2 of 3 | 0.707 | 0.140 |
| X3D-S | 3 of 5 | 0.621 | 0.073 |
| MoViNet-A0 | 1 of 1 | 0.757 | 0.113 |
| MoViNet-A0 | 2 of 2 | 0.621 | 0.027 |
| MoViNet-A0 | 2 of 3 | 0.643 | 0.040 |
| MoViNet-A0 | 3 of 5 | 0.607 | 0.013 |
(note: UCF test mein segments poori video par phaile hain, Pi ke 0.6 s hop se alag; ye rule ka relative asar dikhata hai, per-hour rate nahi.)

### Table 5. Per-category detection (mlp seed 0, Youden thr): recall aur AUC vs normal

X3D-S:
| category | n | recall | AUC vs normal |
|---|---|---|---|
| Abuse | 2 | 0.50 | 0.810 |
| Arrest | 5 | 0.80 | 0.945 |
| Arson | 9 | 1.00 | 0.987 |
| Assault | 3 | 1.00 | 0.969 |
| Burglary | 13 | 0.92 | 0.919 |
| Explosion | 21 | 1.00 | 0.963 |
| Fighting | 5 | 1.00 | 0.973 |
| RoadAccidents | 23 | 0.65 | 0.857 |
| Robbery | 5 | 0.80 | 0.929 |
| Shooting | 23 | 0.91 | 0.946 |
| Shoplifting | 21 | 0.33 | 0.670 |
| Stealing | 5 | 0.40 | 0.708 |
| Vandalism | 5 | 1.00 | 0.971 |

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

### Table 6. Score-level ensemble (X3D + MoViNet, mlp seed 0; offline, 2 backbone = ~2x latency)
| video_auc 0.921 | frame_auc 0.818 | frame_ap 0.234 | prec 0.919 | rec 0.650 | f1 0.762 |

### Table 7. Ablation, X3D-S (mlp-based, seeds 0-2)
| variant | video AUC | frame AUC | frame AP | F1 |
|---|---|---|---|---|
| full | 0.889 ± 0.005 | 0.770 ± 0.008 | 0.182 ± 0.002 | 0.793 ± 0.012 |
| rank_only | 0.891 ± 0.003 | 0.763 ± 0.012 | 0.176 ± 0.008 | 0.812 ± 0.010 |
| topk1 | 0.895 ± 0.009 | 0.779 ± 0.004 | 0.187 ± 0.003 | 0.770 ± 0.013 |
| topk5 | 0.901 ± 0.003 | 0.779 ± 0.011 | 0.183 ± 0.009 | 0.817 ± 0.014 |
| linear_head | 0.888 ± 0.002 | 0.761 ± 0.002 | 0.182 ± 0.003 | 0.791 ± 0.030 |

### Table 7. Ablation, MoViNet-A0 (mlp-based, seeds 0-2)
| variant | video AUC | frame AUC | frame AP | F1 |
|---|---|---|---|---|
| full | 0.920 ± 0.001 | 0.827 ± 0.002 | 0.242 ± 0.005 | 0.821 ± 0.012 |
| rank_only | 0.920 ± 0.003 | 0.822 ± 0.009 | 0.239 ± 0.015 | 0.824 ± 0.010 |
| topk1 | 0.916 ± 0.006 | 0.822 ± 0.002 | 0.241 ± 0.008 | 0.823 ± 0.008 |
| topk5 | 0.922 ± 0.002 | 0.820 ± 0.002 | 0.237 ± 0.004 | 0.832 ± 0.017 |
| linear_head | 0.896 ± 0.000 | 0.780 ± 0.002 | 0.195 ± 0.001 | 0.794 ± 0.038 |

### Table 9. 2D CNN (MobileNetV3-Large, per frame) + temporal head vs 3D backbones (mean ± std over seeds; thr = val Youden)
| model | video AUC | frame AUC | frame AP | precision | recall | F1 | n seeds |
|---|---|---|---|---|---|---|---|
| X3D-S + MLP | 0.889 ± 0.004 | 0.773 ± 0.008 | 0.183 ± 0.005 | 0.809 ± 0.022 | 0.771 ± 0.036 | 0.789 ± 0.011 | 5 |
| MoViNet-A0 + MLP | 0.919 ± 0.003 | 0.828 ± 0.004 | 0.241 ± 0.004 | 0.854 ± 0.026 | 0.781 ± 0.047 | 0.815 ± 0.013 | 5 |
| CNN + lstm | 0.904 ± 0.007 | 0.780 ± 0.004 | 0.168 ± 0.005 | 0.851 ± 0.008 | 0.836 ± 0.044 | 0.842 ± 0.020 | 5 |
| CNN + transformer | 0.894 ± 0.006 | 0.756 ± 0.037 | 0.156 ± 0.013 | 0.864 ± 0.016 | 0.767 ± 0.070 | 0.811 ± 0.038 | 5 |
| CNN + meanpool | 0.906 ± 0.003 | 0.785 ± 0.007 | 0.168 ± 0.005 | 0.853 ± 0.017 | 0.807 ± 0.033 | 0.829 ± 0.011 | 5 |
Paired bootstrap, seed 0, video AUC (B=2000): MoViNet-A0 0.921
- CNN + lstm: 0.895 [0.854, 0.935] | minus MoViNet-A0: -0.025 [-0.060, +0.010] (NOT significant)
- CNN + transformer: 0.891 [0.847, 0.930] | minus MoViNet-A0: -0.030 [-0.076, +0.012] (NOT significant)
- CNN + meanpool: 0.908 [0.870, 0.941] | minus MoViNet-A0: -0.013 [-0.048, +0.022] (NOT significant)
- CNN + lstm operating point (Youden): video TPR 0.836 | video FPR 0.140
- CNN + transformer operating point (Youden): video TPR 0.836 | video FPR 0.120
- CNN + meanpool operating point (Youden): video TPR 0.843 | video FPR 0.153

### Table 8. Stage 2 (X3D-S)
- official 14-class folds, linear: acc 32.6 ± 2.3 (chance 7.1)
- official 14-class folds, mil_topk: acc 34.2 ± 1.2 (chance 7.1)
- DET 13-class (140 anomaly test videos), linear: acc 29.8 ± 0.9 | top-3 60.7 (chance 7.7)
- DET 13-class (140 anomaly test videos), linear_wrs: acc 29.0 ± 1.8 | top-3 57.9 (chance 7.7)
- DET 13-class (140 anomaly test videos), mil_topk: acc 32.4 ± 2.4 | top-3 59.8 (chance 7.7)
- DET 13-class (140 anomaly test videos), mil_topk_wrs: acc 28.8 ± 3.9 | top-3 61.2 (chance 7.7)

Cascade end-to-end (X3D-S; binary mlp seed 0 -> mil_topk stage 2): anomaly videos detected 0.779 | class correct given detected 0.303 | detected AND correct 0.236 | normal test videos falsely alerted 0.200

### Table 8. Stage 2 (MoViNet-A0)
- official 14-class folds, linear: acc 34.1 ± 3.3 (chance 7.1)
- official 14-class folds, mil_topk: acc 36.4 ± 3.0 (chance 7.1)
- DET 13-class (140 anomaly test videos), linear: acc 36.7 ± 1.7 | top-3 62.1 (chance 7.7)
- DET 13-class (140 anomaly test videos), linear_wrs: acc 34.8 ± 3.7 | top-3 60.0 (chance 7.7)
- DET 13-class (140 anomaly test videos), mil_topk: acc 33.3 ± 1.8 | top-3 61.2 (chance 7.7)
- DET 13-class (140 anomaly test videos), mil_topk_wrs: acc 33.3 ± 4.5 | top-3 60.2 (chance 7.7)

Cascade end-to-end (MoViNet-A0; binary mlp seed 0 -> linear stage 2): anomaly videos detected 0.757 | class correct given detected 0.321 | detected AND correct 0.243 | normal test videos falsely alerted 0.113
(figure skip: No module named 'matplotlib' )