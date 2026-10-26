# PiLENS KD2 results (kd2 v1.2, FULL, 2026-10-13) - copied from user's Kaggle log
Official split, thresholds from val only, test once, 5 seeds.

## Binary
| Model | Video AUC | Frame AUC | Frame AP | F1 | Normal FPR |
|---|---|---|---|---|---|
| a0_mlp | 0.919 ± 0.004 | 0.824 ± 0.009 | 0.237 ± 0.007 | 0.831 ± 0.007 | 0.164 ± 0.056 |
| a0_tconv | 0.916 ± 0.009 | 0.850 ± 0.007 | 0.277 ± 0.006 | 0.813 ± 0.024 | 0.145 ± 0.054 |
| a0_tattn | 0.918 ± 0.007 | 0.827 ± 0.015 | 0.276 ± 0.016 | 0.835 ± 0.016 | 0.124 ± 0.023 |
| clip_mlp | 0.968 ± 0.002 | 0.828 ± 0.002 | 0.235 ± 0.004 | 0.904 ± 0.006 | 0.068 ± 0.008 |

Bootstrap: tattn vs tconv frame -0.0106 [-0.0373,+0.0178]; tattn vs mlp +0.0127 [-0.0120,+0.0371]; clip vs a0_mlp frame -0.0034 [-0.045,+0.036], video +0.0435 [+0.0166,+0.0736].

## Cascade A0 alert -> CLIP confirm (+-r segments around A0 peak)
| Gate | r | A0 recall | A0 FPR | Cascade recall | Cascade FPR |
|---|---|---|---|---|---|
| fpr5 | 0 | 63.0 ± 4.5 | 4.7 ± 1.3 | 61.7 ± 4.2 | 1.6 ± 0.5 |
| fpr5 | 1 | 63.0 ± 4.5 | 4.7 ± 1.3 | 62.1 ± 4.4 | 2.0 ± 0.4 |
| fpr5 | 2 | 63.0 ± 4.5 | 4.7 ± 1.3 | 62.3 ± 4.5 | 2.0 ± 0.4 |
| fpr10 | 0 | 76.9 ± 3.5 | 11.6 ± 1.6 | 74.0 ± 4.1 | 2.8 ± 0.5 |
| fpr10 | 1 | 76.9 ± 3.5 | 11.6 ± 1.6 | 74.6 ± 4.0 | 3.9 ± 1.1 |
| fpr10 | 2 | 76.9 ± 3.5 | 11.6 ± 1.6 | 74.4 ± 3.8 | 4.0 ± 1.0 |
| youden | 0 | 83.6 ± 4.7 | 16.4 ± 5.6 | 80.3 ± 3.7 | 3.6 ± 0.9 |
| youden | 1 | 83.6 ± 4.7 | 16.4 ± 5.6 | 81.0 ± 4.4 | 5.1 ± 1.3 |
| youden | 2 | 83.6 ± 4.7 | 16.4 ± 5.6 | 81.1 ± 5.0 | 5.5 ± 1.7 |

## Crime type, 13 classes (gate A0 fpr5)
| Window | A0 cond. | CLIP cond. | A0 e2e | CLIP e2e |
|---|---|---|---|---|
| full video | 35.0 ± 2.5 | 44.0 ± 1.5 | - | - |
| 1 seg | 30.0 ± 2.7 | 35.7 ± 3.7 | 18.9 ± 1.9 | 22.4 ± 2.5 |
| 3 seg | 31.6 ± 2.4 | 34.7 ± 2.4 | 19.9 ± 1.7 | 21.9 ± 1.9 |
| 5 seg | 31.4 ± 2.8 | 33.5 ± 2.3 | 19.7 ± 1.2 | 21.1 ± 2.2 |

## Crime type, 4 groups (gate A0 fpr5)
| Window | A0 cond. | CLIP cond. | A0 e2e | CLIP e2e |
|---|---|---|---|---|
| full video | 63.3 ± 1.9 | 66.0 ± 2.1 | - | - |
| 1 seg | 57.1 ± 4.8 | 57.1 ± 1.9 | 36.1 ± 5.1 | 36.0 ± 3.4 |
| 3 seg | 57.4 ± 5.0 | 58.4 ± 1.5 | 36.3 ± 5.1 | 36.9 ± 3.5 |
| 5 seg | 58.2 ± 6.8 | 61.3 ± 2.1 | 36.9 ± 6.1 | 38.6 ± 2.7 |

Export: CLIP fp32 ONNX 345 MB (diff 6.7e-6), int8 87 MB (cosine 0.981 on random input, accuracy unverified), heads verified.
