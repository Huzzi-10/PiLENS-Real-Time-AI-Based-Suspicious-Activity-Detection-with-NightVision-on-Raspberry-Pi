# PiLENS v3.0: MoViNet-A0 + CLIP alert check

v3 is the current PiLENS. A frozen MoViNet-A0 video model scores the live camera on the Raspberry Pi 5 CPU, and CLIP ViT-B/16 re-checks every alert before it reaches the owner. Compared with v2 (X3D-S) it is smaller, more accurate and faster on the Pi.

This folder is laid out like the PiLENS folder on the Pi itself: the runtime scripts and `export_movinet/` sit at the top, so `v3/` can be copied to the Pi as it is.

## What is in this folder

| Path | What it is |
|---|---|
| `pilens_demo.py` | Live demo with an on-screen panel (FPS, latency, score, alerts), self-test, browser view, recording, GPIO |
| `step6_cascade.py` | Full v3 pipeline: MoViNet-A0 on every clip + CLIP check on every alert, with a per-hour alert report |
| `step6_realtime.py` | Same pipeline without CLIP |
| `step5_pi_benchmark.py` | Pi speed, end-to-end timing and soak (temperature, throttling) test |
| `clip_pi_bench.py` | CLIP speed on the Pi (fp32 and int8) |
| `cascade_samples.sh`, `demo_all.sh` | Run the cascade or the demo over every video in `samples/` |
| `export_movinet/` | MoViNet-A0 model (ONNX backbone 7.8 MB + heads 4.4 MB + `heads_meta.json`) |
| `training/` | Kaggle scripts: feature extraction, heads, ONNX export, CLIP, XD-Violence, A1, night simulation, GPU benchmark, other backbones |
| `evaluation/` | Rebuild the result tables from the saved score packs (numpy only) |
| `results/` | Result summaries and score packs behind every reported number |

## Pipeline settings

| Setting | Value |
|---|---|
| Clip | 13 frames, stride 6 (about 2.4 s at 30 FPS), resize 196, centre crop 172, pixels in [0, 1] |
| Hop | a new clip every 18 frames (0.6 s) |
| Motion gate | clips with less than 0.003 motion are skipped |
| Alert rule | 2 of the last 3 clips above the threshold, then a 10 s cooldown |
| Thresholds (`heads_meta.json`, chosen on validation data) | 0.935 = val FPR 5% (**deployed**), 0.850 = val FPR 10%, 0.779 = Youden |
| CLIP check | 4 frames (0, 4, 8, 12) of the alerted clip at 224 px, binary head + 4-group head |

## Run on the Raspberry Pi 5

One-time setup (64-bit Raspberry Pi OS, camera connected):

```bash
rpicam-hello -t 2000                      # camera check; "no cameras available" -> reboot or reseat the ribbon cable
cd PiLENS/v3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-pi.txt
python pilens_demo.py --check             # PASS / WARN / FAIL list
```

Every new terminal: `cd PiLENS/v3 && source venv/bin/activate`.

```bash
python pilens_demo.py                     # live camera in a window (q = quit)
python pilens_demo.py --web 8000          # watch from a phone or laptop: http://<pi-ip>:8000
python pilens_demo.py --gpio              # LED on GPIO 17, buzzer on GPIO 18
python pilens_demo.py --source samples --record   # every video in samples/, screen recording saved

python step6_cascade.py --dir export_movinet --source camera --max-sec 7200 --quiet   # 2-hour unattended run with CLIP
python step5_pi_benchmark.py --dir export_movinet --soak 10                           # speed + 10-minute soak
```

Prefer `--max-sec` over Ctrl+C: interrupting the camera driver can leave it stuck until a reboot. File names on the Pi are case-sensitive (`Steal.mp4` is not `steal.mp4`).

Each demo run writes `demo_out/<date_time>/` with `report.html`, `report.txt`, `scores.csv`, `fps.csv`, alert snapshots and `summary.png`.

## CLIP model

The CLIP check needs a folder `export_clip_b16/` next to `export_movinet/`:

```
export_clip_b16/
    clip_b16_fp32.onnx        (345 MB, CLIP ViT-B/16 image encoder)
    clip_head_binary.onnx
    clip_head_groups.onnx
    heads_meta_clip.json
```

It is not stored in git because of its size. Get it in one of these ways:

1. Download `export_clip_b16.zip` from the v3.0 entry on the repository's **Releases** page and unzip it inside `v3/`.
2. Or build it: run `training/kd2_kaggle_all.py` on Kaggle and download `download/export_clip_b16.zip` from its output.

The thresholds the Pi uses are in [`results/kd2/heads_meta_clip.json`](results/kd2/heads_meta_clip.json). Use the fp32 model: the int8 file is twice as fast, but its accuracy was never checked on real data.

## Training (Kaggle)

All training runs in Kaggle notebooks (GPU T4, Internet ON, dataset `minmints/ufc-crime-full-dataset`). Each script is pasted into one cell and prints its version first.

| Order | Script | Output |
|---|---|---|
| 0 | `training/m_step0_check.py` | MoViNet loads and exports (no training) |
| 1 | `training/m_step1_extract_movinet.py` | 32 × 2048 features per video |
| 2 | `training/m_step2_binary_mil.py` | binary MIL heads, 5 seeds, `eval_pack.npz` |
| 3 | `training/m_step3_14class.py` | crime-type heads, `cls14_pack.npz` |
| 4 | `training/m_step4_export_onnx.py` | `export_movinet_a0_v2/` (the model in `export_movinet/`) |
| 5 | `training/step8_save_results.py` | `paper_pack.zip` + export zips |
| 6 | `training/kd2_kaggle_all.py` | CLIP heads, cascade results, `export_clip_b16.zip` |
| extra | `training/xd_kaggle_all.py` | XD-Violence results |
| extra | `training/night_groups_a1_kaggle_all.py` | grayscale / simulated-IR models, 4-group head, MoViNet-A1 |
| extra | `training/a1_kaggle_all.py` | MoViNet-A1 alone |
| extra | `training/c_step1/2/5_*.py` | 2D CNN (MobileNetV3) + LSTM / Transformer baselines |
| extra | `training/backbones_*_kaggle.py` | other 3D backbones (scratch / frozen / fine-tuned) |
| extra | `training/gpu_bench.py` | GPU reference speed (Kaggle T4) |
| extra | `training/step7_sample_videos.py` | short real test videos for `samples/` |

## Rebuild the result tables

```bash
python evaluation/paper_eval.py --x3 results/x3ds/binary_out --mv results/movinet_a0_v2/binary_out_movinet_a0_v2 \
    --x3c results/x3ds/cls14_out --mvc results/movinet_a0_v2/cls14_out_movinet_a0_v2 \
    --cnn results/mnv3l/binary_out_mnv3l --out paper_out
python evaluation/reduced_classes_eval.py
```

The first command reproduces [`results/paper_final/paper_tables.md`](results/paper_final/paper_tables.md) exactly.

## Results index

| File | Contents |
|---|---|
| [`results/paper_final/paper_tables.md`](results/paper_final/paper_tables.md) | Main UCF-Crime tables: MoViNet-A0 vs X3D-S vs 2D CNN baselines, CIs, ablations, crime type |
| [`results/paper_final/reduced_classes.md`](results/paper_final/reduced_classes.md) | 4-group and reduced-class crime type |
| [`results/kd2/KD2_SUMMARY.md`](results/kd2/KD2_SUMMARY.md) | CLIP vs A0, cascade (alert check) at three operating points |
| [`results/kd2/PI_CASCADE_LIVE_2H.md`](results/kd2/PI_CASCADE_LIVE_2H.md) | 2-hour live run on the Pi |
| [`results/kd2/PI_CASCADE_SAMPLES.md`](results/kd2/PI_CASCADE_SAMPLES.md) | Cascade on 14 sample clips on the Pi |
| [`results/kd2/PI_CLIP_BENCH.md`](results/kd2/PI_CLIP_BENCH.md) | CLIP speed on the Pi |
| [`results/kd/KD_SUMMARY.md`](results/kd/KD_SUMMARY.md) | Knowledge distillation from CLIP into A0 (did not help) |
| [`results/xd/XD_NOTES.md`](results/xd/XD_NOTES.md) | XD-Violence |
| [`results/night_a1/SUMMARY.md`](results/night_a1/SUMMARY.md) | Grayscale and simulated night IR, 4-group head, MoViNet-A1, Pi speeds |
| [`results/other_backbones/OTHER_BACKBONES.md`](results/other_backbones/OTHER_BACKBONES.md) | Other 3D and 2D backbones (scratch, frozen, fine-tuned) under the same protocol |
| [`results/gpu/GPU_BENCH_T4.md`](results/gpu/GPU_BENCH_T4.md) | GPU reference speeds (Kaggle T4) |

## Notes

- Night: only simulated IR has been tested. Real low-light recordings from the NoIR camera are still needed before any night claim.
- A live run with no alerts gives an upper bound on false alerts per hour, not "zero false alarms".
- Crime type is a hint attached to an alert; it never raises an alert by itself.
- Some console messages and inline code comments are in Roman Urdu (Urdu written in Latin script), as the scripts were first written for the team.
