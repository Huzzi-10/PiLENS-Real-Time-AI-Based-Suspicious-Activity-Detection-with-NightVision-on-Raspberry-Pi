# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Shared helpers for head training on cached features."""

import csv
import json
import random
from pathlib import Path

import numpy as np

from .. import spec


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class FeatureStore:
    """Cached X3D-S features: <dir>/<video_id>.npy (32, 2048) + video_meta.csv."""

    def __init__(self, feats_dir, meta_csv=None, l2norm=False):
        self.dir = Path(feats_dir)
        meta_csv = Path(meta_csv) if meta_csv else self.dir / "video_meta.csv"
        with open(meta_csv, newline="", encoding="utf-8") as f:
            self.meta = {r["video_id"]: r for r in csv.DictReader(f)}
        self.l2norm = l2norm
        self._cache = {}

    def label(self, vid):
        return self.meta[vid]["label"]

    def num_frames(self, vid):
        return int(float(self.meta[vid]["num_frames"]))

    def get(self, vid):
        if vid not in self._cache:
            x = np.load(self.dir / f"{vid}.npy").astype(np.float32)
            if self.l2norm:
                x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
            self._cache[vid] = x
        return self._cache[vid]

    def stack(self, ids):
        return np.stack([self.get(v) for v in ids])

    def check(self, ids):
        missing = [v for v in ids if v not in self.meta or not (self.dir / f"{v}.npy").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} videos have no cached features, e.g. {missing[:5]}")


def class_balanced_sampler(labels, seed):
    """WeightedRandomSampler with weight 1/count(class) per sample."""
    import torch
    from torch.utils.data import WeightedRandomSampler
    labels = np.asarray(labels)
    counts = {c: (labels == c).sum() for c in np.unique(labels)}
    weights = torch.tensor([1.0 / counts[c] for c in labels], dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=g)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def spec_classes():
    return list(spec.CLASSES_14)
