# Reduced-class results (no retraining, from saved logits)

Accuracy %, mean ± std (DET: 3 seeds; folds: 4 folds x 3 seeds). Head: MIL top-k (Pi type); linear head in brackets.

| Task | MoViNet-A0 | X3D-S | Chance |
|---|---|---|---|
| 13 crime classes, top-1 (140 anomaly test videos) | 33.3 ± 1.8 (36.7 ± 1.7) | 32.4 ± 2.4 (29.8 ± 0.9) | 7.7 |
| 13 crime classes, top-3 | 61.2 ± 0.9 (62.1 ± 1.0) | 59.8 ± 0.9 (60.7 ± 0.0) | 23.1 |
| 4 crime groups (140 anomaly test videos) | 59.0 ± 1.2 (59.5 ± 1.8) | 61.2 ± 1.5 (55.2 ± 0.7) | 31.4 (largest group) |
| Official 14 classes incl. Normal (4 folds) | 36.4 ± 3.4 (34.1 ± 3.7) | 34.2 ± 2.9 (32.6 ± 3.2) | 7.1 |
| 4 crime groups + Normal (4 folds) | 59.2 ± 2.2 (56.0 ± 1.8) | 57.6 ± 3.2 (57.4 ± 2.0) | 35.7 (largest group) |
| Notes' 6 classes: Normal, Fighting, Robbery, Shoplifting, Shooting, Abuse (4 folds, 12 videos each) | 50.8 ± 8.0 (51.0 ± 4.2) | 48.5 ± 4.2 (49.2 ± 4.1) | 16.7 |
| Notes' 5 crime classes, without Normal (4 folds) | 55.3 ± 8.4 (56.4 ± 5.2) | 52.9 ± 4.3 (51.5 ± 5.4) | 20.0 |
| Notes' 5 crime classes on the 140-video test set (unbalanced: 23 Shooting, 21 Shoplifting) | 26.2 ± 6.1 (32.7 ± 3.7) | 23.8 ± 6.7 (27.4 ± 1.7) | 20.0 |

DET group sizes {'Violence': 38, 'Theft': 44, 'Fire/damage': 35, 'Road accident': 23}; fold group sizes (Violence, Theft, Fire, Road, Normal) [60, 48, 36, 12, 12] per fold.
