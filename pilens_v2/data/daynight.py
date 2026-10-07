# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Automatic Day/Night label for videos that have none (the Normals).

Night CCTV / IR footage is dark and near-grayscale, so two cheap statistics
separate it well: mean HSV saturation and mean brightness. The thresholds are
NOT hand-picked: they are calibrated on the anomaly videos that already have a
manual Day/Night label, then applied to the unlabeled ones.

    python -m pilens_v2.data.daynight --root /kaggle/input/ufc-crime-full-dataset \
        --index splits/video_index.csv --ann annotations.csv --out splits/day_night.csv

The printed report shows the night share of Normal vs Anomaly videos per split.
If Normals are mostly Day, the model can learn "dark = anomaly" (night shortcut).
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from .annotations import read_events_csv


def frame_stats(path, n_samples=16):
    """Median (saturation, brightness) over uniformly sampled frames, 0..255."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sats, vals = [], []
    for idx in np.linspace(0, max(0, n - 1), n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2HSV)
        sats.append(float(hsv[..., 1].mean()))
        vals.append(float(hsv[..., 2].mean()))
    cap.release()
    if not sats:
        return None
    return float(np.median(sats)), float(np.median(vals))


def predict(stats, sat_thr, val_thr):
    sat, val = stats
    return "Night" if (sat < sat_thr or val < val_thr) else "Day"


def calibrate(labeled):
    """labeled: list of ((sat, val), 'Day'|'Night'). Grid search, balanced accuracy."""
    if not labeled:
        raise ValueError("need at least one labeled video to calibrate")
    sats = np.array([s for (s, _), _ in labeled])
    vals = np.array([v for (_, v), _ in labeled])
    y = np.array([lab == "Night" for _, lab in labeled])
    best = (-1.0, 0.0, 0.0)
    for st in np.unique(np.concatenate([[0.0], np.percentile(sats, np.arange(0, 101, 2))])):
        for vt in np.unique(np.concatenate([[0.0], np.percentile(vals, np.arange(0, 101, 2))])):
            pred = (sats < st) | (vals < vt)
            tpr = (pred & y).sum() / max(1, y.sum())
            tnr = (~pred & ~y).sum() / max(1, (~y).sum())
            bacc = (tpr + tnr) / 2
            if bacc > best[0]:
                best = (float(bacc), float(st), float(vt))
    return {"balanced_acc": best[0], "sat_thr": best[1], "val_thr": best[2]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--index", default="splits/video_index.csv")
    ap.add_argument("--ann", required=True, help="manual annotation CSV with a day_night column")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", default="splits/day_night.csv")
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args()

    with open(args.index, newline="", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    _, meta = read_events_csv(args.ann)

    stats = {}
    for i, row in enumerate(index):
        stats[row["video_id"]] = frame_stats(Path(args.root) / row["rel_path"], args.samples)
        if (i + 1) % 100 == 0:
            print(f"  stats {i + 1}/{len(index)}")

    manual = {v: m["day_night"] for v, m in meta.items() if m["day_night"] in {"Day", "Night"}}
    labeled = [(stats[v], lab) for v, lab in manual.items() if stats.get(v)]
    cal = calibrate(labeled)
    print(f"calibrated on {len(labeled)} labeled videos: {cal}")

    rows, final = [], {}
    for row in index:
        vid, st = row["video_id"], stats[row["video_id"]]
        auto = predict(st, cal["sat_thr"], cal["val_thr"]) if st else ""
        label, source = (manual[vid], "manual") if vid in manual else (auto, "auto")
        final[vid] = label
        rows.append({"video_id": vid, "class": row["label"], "day_night": label, "source": source,
                     "auto_pred": auto, "saturation": f"{st[0]:.2f}" if st else "",
                     "brightness": f"{st[1]:.2f}" if st else ""})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Night-shortcut check per split
    labels = {r["video_id"]: r["label"] for r in index}
    for split_file in sorted(Path(args.splits).glob("anomaly_*.txt")):
        ids = split_file.read_text().split()
        for group in ("Normal", "Anomaly"):
            sel = [v for v in ids if (labels.get(v) == "Normal") == (group == "Normal")]
            night = sum(final.get(v) == "Night" for v in sel)
            print(f"{split_file.stem:15s} {group:8s} night {night}/{len(sel)} "
                  f"({100 * night / max(1, len(sel)):.1f}%)")


if __name__ == "__main__":
    main()
