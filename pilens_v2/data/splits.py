# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Build the video-level split files from the official UCF-Crime lists.

Usage (Kaggle):
    python -m pilens_v2.data.splits \
        --root /kaggle/input/ufc-crime-full-dataset --out splits --probe

What it does:
  * indexes every video under --root and removes byte-identical duplicates
    (the 50 Normal videos that exist twice), mapping every alias to one id;
  * reads the official 14-class 4-fold lists and the anomaly-detection lists,
    skipping empty copies (Anomaly_Detection_splits/Anomaly_Train.txt is empty,
    the root-level one is the real list);
  * carves a stratified, video-level 15% validation set out of each train list;
  * asserts that no video id appears in two splits of the same task.

Commit the resulting ``splits/`` folder to git.
"""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .. import spec

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".mpg", ".mpeg"}
_CLASS_LOOKUP = {c.lower(): c for c in spec.CLASSES_14}
_HASH_CHUNK = 1 << 20  # 1 MiB from the start and end of the file


def video_id_of(path_or_line):
    """'Abuse/Abuse001_x264.mp4' -> 'Abuse001_x264'."""
    name = str(path_or_line).strip().replace("\\", "/").split("/")[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def label_of(video_id):
    """Class name from a UCF-Crime file name; None if it is not recognised."""
    if video_id.lower().startswith("normal"):
        return spec.NORMAL_CLASS
    m = re.match(r"^([A-Za-z]+?)\d", video_id)
    if not m:
        return None
    return _CLASS_LOOKUP.get(m.group(1).lower())


def file_fingerprint(path):
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(_HASH_CHUNK))
        if size > 2 * _HASH_CHUNK:
            f.seek(-_HASH_CHUNK, 2)
            h.update(f.read(_HASH_CHUNK))
    return h.hexdigest()


def index_videos(root):
    """All video files under root, sorted for determinism."""
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file())


def dedupe(paths):
    """Group byte-identical files (same size, same head/tail hash).

    Returns (canonical, alias_to_id, duplicate_rows):
      canonical: video_id -> Path of the kept copy
      alias_to_id: every seen video_id -> canonical video_id
    """
    by_size = defaultdict(list)
    for p in paths:
        by_size[p.stat().st_size].append(p)

    groups = []
    for same_size in by_size.values():
        if len(same_size) == 1:
            groups.append(same_size)
            continue
        by_hash = defaultdict(list)
        for p in same_size:
            by_hash[file_fingerprint(p)].append(p)
        groups.extend(by_hash.values())

    canonical, alias_to_id, dup_rows = {}, {}, []
    for group in groups:
        group = sorted(group, key=lambda p: (len(p.parts), str(p)))
        keep = group[0]
        vid = video_id_of(keep.name)
        if vid in canonical and canonical[vid] != keep:
            # Same name, different content: keep both, disambiguate the second.
            vid = f"{vid}__{keep.parent.name}"
        canonical[vid] = keep
        for p in group:
            alias_to_id.setdefault(video_id_of(p.name), vid)
            if p != keep:
                dup_rows.append({"video_id": vid, "kept": str(keep), "duplicate": str(p)})
    return canonical, alias_to_id, dup_rows


def read_list(path):
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    return [video_id_of(line.split()[0]) for line in lines if line.strip()]


def find_list(root, name, required=True):
    """Find a split file by name, skipping empty copies; prefer the shallowest."""
    candidates = sorted(Path(root).rglob(name), key=lambda p: (len(p.parts), str(p)))
    non_empty = [p for p in candidates if p.stat().st_size > 0 and read_list(p)]
    skipped = [p for p in candidates if p not in non_empty]
    if not non_empty:
        if required:
            raise FileNotFoundError(f"no non-empty {name} under {root} (empty copies: {skipped})")
        return None, skipped
    return non_empty[0], skipped


def stratified_val(ids, labels, frac=0.15, seed=0):
    """Video-level, per-class split of ``ids`` into (train, val)."""
    by_label = defaultdict(list)
    for vid in ids:
        by_label[labels[vid]].append(vid)
    rng = random.Random(seed)
    train, val = [], []
    for lab in sorted(by_label):
        group = sorted(by_label[lab])
        rng.shuffle(group)
        n_val = int(round(frac * len(group)))
        if len(group) >= 2:
            n_val = max(1, n_val)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return sorted(train), sorted(val)


def resolve(ids, alias_to_id, missing):
    out = []
    for vid in ids:
        if vid in alias_to_id:
            out.append(alias_to_id[vid])
        else:
            missing.append(vid)
    return sorted(set(out))


def assert_disjoint(task, parts):
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(parts[a]) & set(parts[b])
            if overlap:
                raise AssertionError(f"{task}: {len(overlap)} videos in both {a} and {b}, e.g. {sorted(overlap)[:3]}")


def probe(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def write_ids(path, ids):
    Path(path).write_text("".join(f"{v}\n" for v in ids), encoding="utf-8")


def build(root, out, val_frac=0.15, seed=0, do_probe=False):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    paths = index_videos(root)
    if not paths:
        raise FileNotFoundError(f"no videos under {root}")
    canonical, alias_to_id, dup_rows = dedupe(paths)
    labels = {vid: label_of(vid) for vid in canonical}
    summary = {"root": str(root), "seed": seed, "val_frac": val_frac,
               "files_found": len(paths), "unique_videos": len(canonical),
               "duplicates_removed": len(dup_rows), "tasks": {}, "warnings": []}

    unlabeled = sorted(v for v, lab in labels.items() if lab is None)
    if unlabeled:
        summary["warnings"].append(f"{len(unlabeled)} videos with unknown class, e.g. {unlabeled[:5]}")

    # --- video index -------------------------------------------------------
    with open(out / "video_index.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["video_id", "label", "rel_path", "size_bytes", "num_frames", "fps", "width", "height"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for vid in sorted(canonical):
            p = canonical[vid]
            row = {"video_id": vid, "label": labels[vid] or "",
                   "rel_path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size}
            if do_probe:
                row["num_frames"], row["fps"], row["width"], row["height"] = probe(p)
            w.writerow(row)
    with open(out / "duplicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "kept", "duplicate"])
        w.writeheader()
        w.writerows(dup_rows)

    # --- Task A: 14-class, official 4 folds -------------------------------
    task_a = {}
    for k in range(1, 5):
        tr_file, tr_skip = find_list(root, f"train_{k:03d}.txt", required=False)
        te_file, te_skip = find_list(root, f"test_{k:03d}.txt", required=False)
        if tr_file is None or te_file is None:
            summary["warnings"].append(f"14-class fold {k} lists not found")
            continue
        missing = []
        train_all = resolve(read_list(tr_file), alias_to_id, missing)
        test = resolve(read_list(te_file), alias_to_id, missing)
        train, val = stratified_val(train_all, labels, val_frac, seed)
        assert_disjoint(f"cls14 fold {k}", {"train": train, "val": val, "test": test})
        for part, ids in (("train", train), ("val", val), ("test", test)):
            write_ids(out / f"cls14_fold{k}_{part}.txt", ids)
        task_a[f"fold{k}"] = {
            "train_list": str(tr_file), "test_list": str(te_file),
            "skipped_empty": [str(p) for p in tr_skip + te_skip],
            "train": len(train), "val": len(val), "test": len(test),
            "test_per_class": dict(Counter(labels[v] for v in test)),
            "missing_from_disk": missing,
        }
    summary["tasks"]["cls14"] = task_a

    # --- Task B: anomaly detection (1610 / 290) ---------------------------
    tr_file, tr_skip = find_list(root, "Anomaly_Train.txt")
    te_file, te_skip = find_list(root, "Anomaly_Test.txt")
    missing = []
    train_all = resolve(read_list(tr_file), alias_to_id, missing)
    test = resolve(read_list(te_file), alias_to_id, missing)
    is_anom = {v: ("Anomaly" if labels[v] != spec.NORMAL_CLASS else "Normal") for v in train_all}
    # stratify on the fine class so every anomaly class is in val
    train, val = stratified_val(train_all, {v: labels[v] or is_anom[v] for v in train_all}, val_frac, seed)
    assert_disjoint("anomaly", {"train": train, "val": val, "test": test})
    for part, ids in (("train", train), ("val", val), ("test", test)):
        write_ids(out / f"anomaly_{part}.txt", ids)
    summary["tasks"]["anomaly"] = {
        "train_list": str(tr_file), "test_list": str(te_file),
        "skipped_empty": [str(p) for p in tr_skip + te_skip],
        "train": len(train), "val": len(val), "test": len(test),
        "train_normal": sum(labels[v] == spec.NORMAL_CLASS for v in train),
        "val_normal": sum(labels[v] == spec.NORMAL_CLASS for v in val),
        "test_normal": sum(labels[v] == spec.NORMAL_CLASS for v in test),
        "missing_from_disk": missing,
    }

    (out / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root (searched recursively)")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true", help="also read frame count / fps with OpenCV")
    args = ap.parse_args()
    s = build(args.root, args.out, args.val_frac, args.seed, args.probe)
    print(json.dumps({k: v for k, v in s.items() if k != "tasks"}, indent=2))
    for task, info in s["tasks"].items():
        print(task, json.dumps(info, indent=1)[:2000])


if __name__ == "__main__":
    main()
