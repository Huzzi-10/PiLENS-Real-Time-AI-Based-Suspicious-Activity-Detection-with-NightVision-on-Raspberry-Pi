# PiLENS v2.0: X3D-S two-stage cascade

> Replaced by [v3](../v3/), which uses MoViNet-A0 and is both more accurate (video AUC 0.919 vs 0.889 on UCF-Crime) and faster on the Pi (137 vs 196 ms per clip). v2 is kept so its results stay reproducible. Run every command below from this `v2/` folder.

v2 replaced the v1 ResNet18-LSTM classifier with a frozen X3D-S backbone and a weakly supervised MIL head that scores every clip, plus a 14-class UCF-Crime head that runs only on alerts. Everything is exported to ONNX Runtime for the Raspberry Pi 5.

| Path | What it is |
|---|---|
| `pilens_v2/` | Python package: data splits, features, MIL and 14-class heads, ONNX export, Pi runtime, benchmark, evaluation |
| `notebooks/PiLENS_v2_Kaggle.ipynb` | Splits, features, training and ONNX export in one Run All |
| `scripts/` | Pi setup, all benchmarks, systemd service |
| `tests/` | Unit and end-to-end tests (`python -m pytest tests`) |
| `splits/` | Split files |
| `docs/` | Engineering notes, quickstart, result tables, supervisor and ethics documents |

X3D-S results under the v3 protocol (5 seeds, thresholds from validation): video AUC 0.889 ± 0.004, frame AUC 0.773; Pi 5, 3 threads: 196 ms per clip. See [`../v3/results/paper_final/paper_tables.md`](../v3/results/paper_final/paper_tables.md).

Start here: [docs/v2_quickstart.md](docs/v2_quickstart.md).
