# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Runtime building blocks shared by the live pipeline and offline replay:
frame ring buffer, motion gate, k-of-n voter and the ONNX cascade."""

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import spec
from ..preprocess import clip_timestamps, frames_to_clip


# --- ring buffer -------------------------------------------------------------

class FrameRing:
    """Thread-safe (timestamp, frame) ring. Appends never block on readers."""

    def __init__(self, seconds, fps):
        self._buf = deque(maxlen=max(spec.CLIP_LEN, int(np.ceil(seconds * fps)) + 1))
        self._lock = threading.Lock()

    def append(self, t, frame):
        with self._lock:
            self._buf.append((t, frame))

    def snapshot(self):
        with self._lock:
            return list(self._buf)

    def latest(self):
        with self._lock:
            return self._buf[-1] if self._buf else (None, None)

    def span(self):
        with self._lock:
            return (self._buf[-1][0] - self._buf[0][0]) if len(self._buf) > 1 else 0.0

    def clip_at(self, t_end):
        """CLIP_LEN frames nearest to the clip's target timestamps ending at t_end."""
        items = self.snapshot()
        if not items:
            return None
        times = np.array([t for t, _ in items])
        targets = clip_timestamps(t_end)
        if targets[0] < times[0] - spec.FRAME_STEP_SEC:
            return None  # buffer does not cover the clip yet
        idx = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
        prev = np.clip(idx - 1, 0, len(times) - 1)
        idx = np.where(np.abs(times[prev] - targets) < np.abs(times[idx] - targets), prev, idx)
        return [items[i][1] for i in idx]

    def since(self, t_from):
        return [(t, f) for t, f in self.snapshot() if t >= t_from]


# --- motion gate --------------------------------------------------------------

class MotionGate:
    """MOG2 on a small blurred gray frame. Cheapest stage: no motion -> no model.

    A blob counts only if its area >= min_area_frac of the frame and it
    persists for ``persist`` consecutive checks (filters insects near the IR
    LED, flicker, single-frame noise). Once active, the gate stays open for
    ``hold_sec`` so the clip that covers the event is still scored.
    """

    def __init__(self, size=(320, 180), min_area_frac=0.002, persist=3, hold_sec=3.0,
                 history=300, var_threshold=25, temporal_avg=2):
        import cv2
        self.cv2 = cv2
        self.size = size
        self.min_area = min_area_frac * size[0] * size[1]
        self.persist = persist
        self.hold_sec = hold_sec
        self.mog = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                      detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.recent = deque(maxlen=max(1, temporal_avg))
        self.streak = 0
        self.last_motion_t = -np.inf
        self.motion_frac = 0.0
        self.box = None

    def update(self, frame_bgr, t):
        cv2 = self.cv2
        small = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        self.recent.append(gray.astype(np.float32))
        gray = (sum(self.recent) / len(self.recent)).astype(np.uint8)  # 2-3 frame temporal average
        mask = self.mog.apply(gray)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadows (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        self.motion_frac = float(sum(cv2.contourArea(c) for c in big)) / (self.size[0] * self.size[1])
        if big:
            self.streak += 1
            pts = np.concatenate(big)
            self.box = cv2.boundingRect(pts)  # union box in gate coordinates (crop ablation)
        else:
            self.streak = 0
        if self.streak >= self.persist:
            self.last_motion_t = t
        return self.active(t)

    def active(self, t):
        return (t - self.last_motion_t) <= self.hold_sec


# --- k-of-n voter ---------------------------------------------------------------

class Voter:
    """Alert when >= k of the last n clips are suspicious, then cool down."""

    def __init__(self, k=spec.VOTE_K, n=spec.VOTE_N, cooldown_sec=30.0):
        self.k, self.n, self.cooldown = k, n, cooldown_sec
        self.window = deque(maxlen=n)
        self.last_alert = -np.inf

    def push(self, suspicious, t):
        self.window.append(bool(suspicious))
        if sum(self.window) >= self.k and (t - self.last_alert) >= self.cooldown:
            self.last_alert = t
            self.window.clear()
            return True
        return False


# --- ONNX cascade ------------------------------------------------------------

@dataclass
class Prediction:
    label: str
    top3: list = field(default_factory=list)  # [(class, prob), ...]


class Cascade:
    """backbone.onnx once per clip; binary head every clip; cls14 head on alert."""

    def __init__(self, exports="exports", threads=4, backbone="backbone.onnx",
                 threshold=None, min_conf=spec.STAGE2_MIN_CONF, history=8):
        import onnxruntime as ort
        ex = Path(exports)
        self.meta = json.loads((ex / "pilens_v2.json").read_text())
        if self.meta.get("spec", {}).get("clip_len") != spec.CLIP_LEN:
            raise ValueError("exports were made with a different clip spec")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        mk = lambda name: ort.InferenceSession(str(ex / name), so, providers=["CPUExecutionProvider"])
        self.bb, self.bin, self.cls = mk(backbone), mk("binary_head.onnx"), mk("cls14_head.onnx")
        self.gray3 = bool(self.meta.get("gray3", False))
        self.threshold = float(threshold if threshold is not None else self.meta.get("threshold", 0.5))
        self.classes = self.meta["spec"]["classes_14"]
        self.min_conf = min_conf
        self.feats = deque(maxlen=max(history, spec.STAGE2_CLIPS, spec.VOTE_N))

    def preprocess(self, frames):
        return frames_to_clip(frames, self.gray3)[None]

    def step(self, clip):
        """One clip -> (suspicious score of this clip, ms backbone, ms head)."""
        import time
        t0 = time.perf_counter()
        feat = self.bb.run(None, {"clip": clip})[0][0]
        t1 = time.perf_counter()
        self.feats.append(feat)
        ctx = np.stack(list(self.feats)[-spec.VOTE_N:])[None].astype(np.float32)
        score = float(self.bin.run(None, {"feats": ctx})[0][0, -1])
        t2 = time.perf_counter()
        return score, (t1 - t0) * 1e3, (t2 - t1) * 1e3

    def classify(self):
        """Stage 2 on the average context of the last STAGE2_CLIPS cached features."""
        k = min(len(self.feats), spec.STAGE2_CLIPS)
        if k < 3:  # top-k heads need >= 3 clips; repeat the last ones
            ctx = np.stack(([self.feats[-1]] * (3 - k)) + list(self.feats)[-k:])
        else:
            ctx = np.stack(list(self.feats)[-k:])
        logits = self.cls.run(None, {"feats": ctx[None].astype(np.float32)})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = [i for i in np.argsort(-p) if self.classes[i] != spec.NORMAL_CLASS][:3]
        top3 = [(self.classes[i], float(p[i])) for i in order]
        label = top3[0][0] if top3 and top3[0][1] >= self.min_conf else spec.UNKNOWN_LABEL
        return Prediction(label, top3)
