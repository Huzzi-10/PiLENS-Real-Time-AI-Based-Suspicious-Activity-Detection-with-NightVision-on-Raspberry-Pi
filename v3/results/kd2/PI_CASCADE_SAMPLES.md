# Pi 5 live cascade on 14 sample clips (2026-10-13, step6_cascade v1.0, A0 export_movinet + CLIP fp32 3 thr, real-time pacing)
Video-level (any alert): 
| op | A0 alone: anomaly clips alerted / 9 | A0+CLIP / 9 | normal clips alerted / 5 |
| youden 0.779 | 7 (Burglary018, Explosion025, Fighting018, Shooting037, Shoplifting017, fight, Steal) | 6 (Explosion025 rejected, p=1.8e-8) | 0 / 0 |
| fpr10 0.850 | 6 | 5 | 0 / 0 |
| fpr5 0.935 (deployed) | 6 | 5 | 0 / 0 |
Missed by A0 at all ops: RoadAccidents021 (max 0.31), shoplifting.mp4 (max 0.17).
CLIP rejected NO false alarm here (normal clips gave 0 A0 alerts); it removed one TRUE alert (Explosion025).
CLIP confirm time live, A0 running in parallel (3+3 threads on 4 cores): 2.0-4.8 s, typical ~3.4 s (vs 1.8 s alone in kd_pi_bench).
CLIP group hint: Burglary/Shoplifting/Steal -> Theft, Fighting018 -> Violence, fight.mp4 -> Fire/damage (wrong), Shooting037 -> Road accident (wrong), Explosion -> Fire/damage (right, though binary rejected).
Anecdotal (14 clips); per-hour numbers in the output are meaningless for short clips.
Camera 2 h run failed: "no cameras available" (libcamera) -> hardware/cable/reboot issue, rerun pending.
