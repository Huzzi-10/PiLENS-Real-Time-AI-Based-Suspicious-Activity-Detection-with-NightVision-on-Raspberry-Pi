# PILENS_FINAL run (2026-10-12): gray / simulated-IR / 4-group head / MoViNet-A1
Official split, 5 seeds, val-chosen thresholds, test once. MLP head = Pi head.

## Binary (video AUC mean+-std; frame AUC)
| Model (train) | test colour | test gray | test simulated IR |
|---|---|---|---|
| A0 colour | 0.919+-0.003 (frame 0.828) | 0.869 (0.788) | 0.750 (0.708) |
| A0 gray   | -                           | 0.902+-0.006 (0.801) | 0.774 (0.710) |
| A0 gray, tconv head | - | 0.911+-0.006 (0.823) | 0.733 (0.704) |
| A1 colour | 0.928+-0.004 (frame 0.804), tconv 0.922 (0.826) | - | - |
Paired bootstrap (290 test videos, seed-mean score): A1-A0 +0.011 [-0.014,+0.035] (not significant); gray-on-gray - colour-on-colour -0.017 [-0.041,+0.006]; gray model vs colour model on gray test +0.032 [+0.009,+0.057]; on simulated IR +0.035 [+0.005,+0.066].
Simulated IR FPR at val threshold: colour model 35%, gray model 17%.
A1: precision 0.909 vs 0.854, F1 0.841 vs 0.815, recall 0.783 vs 0.781; frame AUC lower than A0.

## Crime type (MIL top-k head; linear in brackets)
| Task | A0 colour | A0 gray | A1 |
|---|---|---|---|
| official 14 classes (folds) | 36.4 (34.1) | 33.1 (33.0) | 35.2 (34.7) |
| 13 classes, 140 crime test videos | 33.3 (36.7) | 34.0 (35.2) | 36.9 (39.3) |
| dedicated 4-group head, 140 test videos | 62.4 (60.5) | 59.0 (55.7) | 62.1 (57.9) |
(earlier 4 groups from 13-class logits: 59.0). Chance 31.4.
A1 export 13.8 MB backbone + 4.4 MB heads; A0 gray 7.8 MB + 4.4 MB. Pi benchmark for A1 still pending.

## Pi 5 benchmark (2026-10-12, onnxruntime 1.30, 3 threads best)
| Model | backbone ms/clip (median) | 10-min soak | temp | throttling |
|---|---|---|---|---|
| MoViNet-A0 (colour) | 137 | 135 flat | 65-67 C | none |
| MoViNet-A0 gray | 136 | 134-136 flat | 63-67 C | none |
| MoViNet-A1 | 256 (p95 276) | 254 flat | 63-67 C | none |
Hop budget 0.6 s: A0 4.4x margin, A1 2.3x margin. A1 ~1.9x slower for +0.011 video AUC (not significant).
