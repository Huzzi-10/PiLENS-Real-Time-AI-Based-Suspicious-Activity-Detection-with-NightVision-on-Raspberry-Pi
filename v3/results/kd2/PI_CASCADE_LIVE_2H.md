# Pi 5 live 2-hour normal-room run with cascade (2026-10-13/14, step6_cascade v1.0, A0 export_movinet + CLIP fp32 3 thr)
7200 s | frames 216036 (30.0 FPS) | clips 11998 | motion-gated 7945 (66%) | dropped 0 | max temp 63.9 C
per-clip compute median 168 / p95 246 ms; clip-end to decision median 170 / p95 248 ms (CLIP session loaded in same process)
| op | A0 thr | A0 alerts | A0+CLIP | A0 per h | cascade per h |
| youden | 0.779 | 1 | 0 | 0.50 | 0 (95% upper 1.50) |
| fpr10 | 0.850 | 0 | 0 | 0 (upper 1.50) | 0 (upper 1.50) |
| fpr5 (deployed) | 0.935 | 0 | 0 | 0 (upper 1.50) | 0 (upper 1.50) |
Single A0 alert at t=5808 s, score 0.831 (A0 type hint Robbery 0.64); CLIP p=8.3e-7 -> rejected (correct, room was normal); CLIP time 3627 ms.
Max A0 score over 2 h: 0.863 (below fpr10/fpr5 thresholds).
Paper: "2 h live normal scene: 0 alerts at the deployed point (95% upper bound 1.5 FA/h); at the sensitive Youden point A0 raised 1 false alert, which CLIP rejected." Do not claim 0 FA/h. Scene activity level not logged beyond 66% gated clips.
