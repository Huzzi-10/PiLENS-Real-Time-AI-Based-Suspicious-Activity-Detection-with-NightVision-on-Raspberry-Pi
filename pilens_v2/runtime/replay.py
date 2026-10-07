# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Offline, deterministic replay of test videos through the SAME runtime code
(ring buffer, motion gate, ONNX cascade) to get per-hop scores in video time.

    python -m pilens_v2.runtime.replay --exports exports --root <dataset root> \
        --index splits/video_index.csv --ids splits/anomaly_test.txt --out replay/test

Writes <out>/<video_id>.csv with columns t, motion, score (score is always
computed, so the motion gate can be switched on/off afterwards). Then:

    python -m pilens_v2.eval.alert_eval --replay replay/test --index splits/video_index.csv \
        --ann Temporal_Anomaly_Annotation_for_Testing_Videos.txt --threshold <val thr>

gives event-to-alert time, missed events and false alarms per hour for each
k-of-n rule (cascade / "2 of 3" ablation table).
"""

import argparse
import csv
from pathlib import Path

from .. import spec
from .core import Cascade, FrameRing, MotionGate


def replay_video(path, cascade, hop=0.6, motion_hz=10.0):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or spec.SOURCE_FPS
    ring = FrameRing(spec.CLIP_SPAN_SEC + 1.0, fps)
    gate = MotionGate()
    cascade.feats.clear()
    rows, n, next_hop, next_motion = [], 0, spec.CLIP_SPAN_SEC, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        n += 1
        ring.append(t, frame)
        if t >= next_motion:
            gate.update(frame, t)
            next_motion += 1.0 / motion_hz
        if t >= next_hop:
            next_hop += hop
            frames = ring.clip_at(t)
            if frames is None:
                continue
            score, _, _ = cascade.step(cascade.preprocess(frames))
            rows.append({"t": round(t, 3), "motion": int(gate.active(t)), "score": round(score, 5)})
    cap.release()
    return rows, n, fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", default="exports")
    ap.add_argument("--backbone", default="backbone.onnx")
    ap.add_argument("--root", required=True)
    ap.add_argument("--index", default="splits/video_index.csv")
    ap.add_argument("--ids", required=True, help="split file with video ids")
    ap.add_argument("--hop", type=float, default=0.6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="replay/test")
    args = ap.parse_args()

    with open(args.index, newline="", encoding="utf-8") as f:
        index = {r["video_id"]: r for r in csv.DictReader(f)}
    ids = Path(args.ids).read_text().split()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cascade = Cascade(args.exports, args.threads, args.backbone)
    for i, vid in enumerate(ids):
        dst = out / f"{vid}.csv"
        if dst.exists():
            continue
        rows, n, fps = replay_video(Path(args.root) / index[vid]["rel_path"], cascade, args.hop)
        with open(dst, "w", newline="") as f:
            f.write(f"# frames={n} fps={fps}\n")
            w = csv.DictWriter(f, fieldnames=["t", "motion", "score"])
            w.writeheader()
            w.writerows(rows)
        print(f"{i + 1}/{len(ids)} {vid}: {len(rows)} hops")


if __name__ == "__main__":
    main()
