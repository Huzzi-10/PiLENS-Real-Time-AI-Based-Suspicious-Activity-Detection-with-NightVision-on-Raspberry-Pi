# Changelog

All notable changes to PiLENS, one entry per version. Versions follow `MAJOR.MINOR.PATCH`: a new model or pipeline is a new major version, new features a minor version, fixes a patch.

## [Unreleased]

## [3.0.0] - 2026-10-26

Current version, in [`v3/`](v3/).

### Added
- MoViNet-A0 (Kinetics-600, frozen, 3.1M parameters) as the video backbone, with a weakly supervised MIL head. UCF-Crime video AUC 0.919 ± 0.003 over 5 seeds.
- CLIP ViT-B/16 alert check: runs only when an alert is raised and confirms or rejects it. On the UCF-Crime test set it cut false alerts on normal videos from 16.4% to 3.6% (recall 83.6% to 80.3%).
- Motion gate, 2-of-3 alert rule, 10 s cooldown and a 4-group crime-type hint (violence, theft, fire or damage, road accident).
- Pi runtime and tools: `pilens_demo.py` (live panel, self-test, browser view, recording, GPIO), `step6_cascade.py`, `step6_realtime.py`, `step5_pi_benchmark.py`, `clip_pi_bench.py`.
- MoViNet-A0 ONNX model in `v3/export_movinet/`.
- Kaggle training scripts for every experiment, and evaluation scripts that rebuild the result tables from saved score packs.
- Second dataset: XD-Violence (video AUC 0.980 on the 753-video single-label test subset).
- Comparisons under one protocol: X3D-S, MoViNet-A1, 2D CNN + LSTM / Transformer, and other 3D backbones; grayscale and simulated night IR; GPU reference speeds.
- Pi measurements: 137 ms per clip; 2-hour live run at 30 FPS with 0 dropped frames and a maximum of 63.9 °C.

### Changed
- Repository reorganised by version: `v1/`, `v2/`, `v3/`. Shared photos stay in `assets/`, wiring notes in `docs/hardware/`.

### Removed
- The v2 paper draft (`docs/paper/`), which had placeholder numbers and an outdated design. It is still in the `v2.0.0` tag.

## [2.0.0] - 2026-10-07

In [`v2/`](v2/).

### Added
- `pilens_v2` package: X3D-S backbone, MIL anomaly head, 14-class head on alerts, ONNX export, Pi runtime, benchmarks, tests and a Kaggle Run-All notebook.

## [1.0.0] - 2026-07-14

In [`v1/`](v1/).

### Added
- First prototype: YOLO11n person detection, ResNet18-LSTM activity classifier, face recognition mode, PyQt6 GUI, Flask MJPEG stream, Tailscale access, e-mail, LED and buzzer alerts.

[3.0.0]: https://github.com/Huzzi-10/PiLENS-Real-Time-AI-Based-Suspicious-Activity-Detection-with-NightVision-on-Raspberry-Pi/tree/v3.0.0
[2.0.0]: https://github.com/Huzzi-10/PiLENS-Real-Time-AI-Based-Suspicious-Activity-Detection-with-NightVision-on-Raspberry-Pi/tree/v2.0.0
[1.0.0]: https://github.com/Huzzi-10/PiLENS-Real-Time-AI-Based-Suspicious-Activity-Detection-with-NightVision-on-Raspberry-Pi/tree/v1.0.0
