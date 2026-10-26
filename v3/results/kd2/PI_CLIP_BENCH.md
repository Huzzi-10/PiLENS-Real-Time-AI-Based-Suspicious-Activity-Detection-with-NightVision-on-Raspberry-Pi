# Pi 5 CLIP ViT-B/16 benchmark (2026-10-13, kd_pi_bench v1.0, ORT 1.30)
Model size: fp32 345 MB, int8 87 MB.
4 frames (= 1 clip confirm, cascade r=0): fp32 best 4 thr 1774 ms (3 thr 2003, p95 2029); int8 4 thr 773 ms (3 thr 791, p95 810).
1 frame: fp32 404 ms (4 thr), int8 157 ms (4 thr).
Alert window: 1 clip fp32 1756 / int8 763; 3 clips 6466 / 2808; 5 clips 11385 / 6076 ms.
Soak 3 min fp32 4thr 4 frames: 1754-3072 ms (median ~1.8 s, one spike), 68.8-72.2 C, 2400 MHz, throttled=0x0.
Notes: int8 accuracy NOT verified on real data (only cosine 0.981 on random input) -> paper uses fp32 (accuracy measured).
CLIP runs only on alerts, so average load unchanged; during confirm it competes with A0 for CPU (not measured live).
