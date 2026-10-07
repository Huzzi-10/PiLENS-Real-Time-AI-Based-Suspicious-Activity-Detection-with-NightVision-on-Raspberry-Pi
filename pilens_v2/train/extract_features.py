# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Step 2 (B1): cache X3D-S features, 32 clips per untrimmed video.

Quick test first, then the full run (resumes, skips finished videos):
    python -m pilens_v2.train.extract_features --root /kaggle/input/ufc-crime-full-dataset \
        --index splits/video_index.csv --out /kaggle/working/feats_x3ds --limit 8
    python -m pilens_v2.train.extract_features ... --shard 0/2 --device cuda:0 &
    python -m pilens_v2.train.extract_features ... --shard 1/2 --device cuda:1 &
    python -m pilens_v2.train.extract_features ... --validate-only

Output: <out>/<video_id>.npy  (32, 2048) float16 and <out>/video_meta.csv.
At the end it prints "valid: N/N"; only then save the Kaggle output as a dataset.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .. import spec
from ..preprocess import frames_to_clip, segment_clip_indices


def read_clips(path, num_frames, gray3=False, crop_mode="center"):
    """(N_SEGMENTS, 3, T, H, W) float32 for one video, one seek per clip."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if num_frames <= 0:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_idx = segment_clip_indices(num_frames)
    clips, last_good = [], None
    for idx in all_idx:
        lo, hi = int(idx.min()), int(idx.max())
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        span = {}
        for f in range(lo, hi + 1):
            ok, frame = cap.read()
            if not ok:
                break
            span[f] = frame
            last_good = frame
        frames = []
        for i in idx:
            fr = span.get(int(i))
            if fr is None:  # decode ended early: reuse the nearest decoded frame
                fr = span[max(span)] if span else last_good
            if fr is None:
                raise IOError(f"could not decode any frame from {path}")
            frames.append(fr)
        clips.append(frames_to_clip(frames, gray3, crop_mode))
    cap.release()
    return np.stack(clips), num_frames


def validate(out_dir, ids):
    ok, bad = 0, []
    for vid in ids:
        p = Path(out_dir) / f"{vid}.npy"
        try:
            a = np.load(p)
            if a.shape == (spec.N_SEGMENTS, spec.FEATURE_DIM) and np.isfinite(a).all():
                ok += 1
                continue
            bad.append((vid, f"shape {a.shape} or non-finite"))
        except Exception as e:  # missing or corrupt
            bad.append((vid, str(e)))
    return ok, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--index", default="splits/video_index.csv")
    ap.add_argument("--out", default="feats_x3ds")
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (quick test)")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th video starting at i")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16, help="clips per forward pass")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gray3", action="store_true", help="grayscale -> 3ch preprocessing")
    ap.add_argument("--crop-mode", default="center", choices=["center", "squash"])
    ap.add_argument("--backbone", default="x3d_s", choices=["x3d_s", "tiny"])
    ap.add_argument("--weights", default=None, help="local X3D_S.pyth (offline)")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    with open(args.index, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.validate_only:
        import torch
        from torch.utils.data import DataLoader, Dataset

        from ..models.backbone import build_backbone

        i, n = (int(x) for x in args.shard.split("/"))
        todo = [r for k, r in enumerate(rows) if k % n == i and not (out / f"{r['video_id']}.npy").exists()]
        print(f"shard {i}/{n}: {len(todo)} videos to extract")

        class Videos(Dataset):
            def __len__(self):
                return len(todo)

            def __getitem__(self, k):
                r = todo[k]
                nf = int(r["num_frames"]) if r.get("num_frames") else 0
                clips, nf = read_clips(Path(args.root) / r["rel_path"], nf, args.gray3, args.crop_mode)
                return r["video_id"], torch.from_numpy(clips), nf

        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        model = build_backbone(args.backbone, weights=args.weights).to(device)
        loader = DataLoader(Videos(), batch_size=None, num_workers=args.workers)
        meta_path = out / f"video_meta_shard{i}.csv"
        new_file = not meta_path.exists()
        t0 = time.time()
        with open(meta_path, "a", newline="", encoding="utf-8") as mf, torch.no_grad():
            w = csv.writer(mf)
            if new_file:
                w.writerow(["video_id", "num_frames"])
            for k, (vid, clips, nf) in enumerate(loader):
                feats = torch.cat([model(clips[s:s + args.batch].to(device)).float().cpu()
                                   for s in range(0, len(clips), args.batch)])
                np.save(out / f"{vid}.npy", feats.numpy().astype(np.float16))
                w.writerow([vid, nf])
                mf.flush()
                if (k + 1) % 10 == 0 or k + 1 == len(todo):
                    rate = (time.time() - t0) / (k + 1)
                    print(f"  {k + 1}/{len(todo)}  {rate:.1f}s/video  eta {rate * (len(todo) - k - 1) / 60:.0f} min")

    # merge per-shard meta with the index, then validate everything
    nf = {}
    for p in sorted(out.glob("video_meta_shard*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            nf.update({r["video_id"]: r["num_frames"] for r in csv.DictReader(f)})
    with open(out / "video_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "num_frames", "rel_path"])
        for r in rows:
            if r["video_id"] in nf:
                w.writerow([r["video_id"], r["label"], nf[r["video_id"]], r["rel_path"]])
    ok, bad = validate(out, [r["video_id"] for r in rows])
    print(f"valid: {ok}/{len(rows)}")
    for vid, err in bad[:20]:
        print("  BAD", vid, err)


if __name__ == "__main__":
    main()
