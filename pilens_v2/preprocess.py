# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Clip preprocessing shared by feature extraction, ONNX parity and the Pi.

Only numpy + OpenCV, so the exact same code runs on Kaggle and on the Pi.
"""

import cv2
import numpy as np

from . import spec

CROP_MODES = ("center", "squash")


def to_gray3(frame_bgr):
    """Grayscale replicated to 3 channels (night-vision / NoIR consistency)."""
    if frame_bgr.ndim == 2:
        gray = frame_bgr
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.merge([gray, gray, gray])


def resize_short_side(frame, short_side=spec.SHORT_SIDE):
    h, w = frame.shape[:2]
    if h <= w:
        new_h, new_w = short_side, int(round(w * short_side / h))
    else:
        new_h, new_w = int(round(h * short_side / w)), short_side
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def center_crop(frame, size=spec.CROP_SIZE):
    h, w = frame.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return frame[top:top + size, left:left + size]


def prepare_frame(frame_bgr, gray3=False, crop_mode="center"):
    """One BGR (or gray) uint8 frame -> RGB uint8 CROP_SIZE x CROP_SIZE x 3.

    crop_mode="center": resize short side to 182, center crop 160 (default,
    cuts ~1/3 of a 4:3 frame's width). crop_mode="squash": resize the full
    frame to 160x160 so edge events are kept (crop ablation).
    """
    if crop_mode not in CROP_MODES:
        raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}")
    if gray3 or frame_bgr.ndim == 2:
        frame_bgr = to_gray3(frame_bgr)
    if crop_mode == "center":
        out = center_crop(resize_short_side(frame_bgr))
    else:
        out = cv2.resize(frame_bgr, (spec.CROP_SIZE, spec.CROP_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def frames_to_clip(frames_bgr, gray3=False, crop_mode="center"):
    """List of CLIP_LEN frames -> float32 array (3, T, H, W), normalised."""
    if len(frames_bgr) != spec.CLIP_LEN:
        raise ValueError(f"expected {spec.CLIP_LEN} frames, got {len(frames_bgr)}")
    rgb = np.stack([prepare_frame(f, gray3, crop_mode) for f in frames_bgr])  # T,H,W,3
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.asarray(spec.MEAN, np.float32)) / np.asarray(spec.STD, np.float32)
    return np.ascontiguousarray(x.transpose(3, 0, 1, 2))  # 3,T,H,W


# --- Temporal sampling ------------------------------------------------------

def clip_indices(center, num_frames, clip_len=spec.CLIP_LEN, stride=spec.FRAME_STRIDE):
    """Frame indices of a clip centred on ``center`` (clamped to the video)."""
    offsets = (np.arange(clip_len) - (clip_len - 1) / 2.0) * stride
    idx = np.round(center + offsets).astype(int)
    return np.clip(idx, 0, max(0, num_frames - 1))


def segment_bounds(num_frames, n_segments=spec.N_SEGMENTS):
    """[start, end) frame bounds of ``n_segments`` equal temporal segments."""
    edges = np.linspace(0, num_frames, n_segments + 1)
    starts = np.floor(edges[:-1]).astype(int)
    ends = np.maximum(np.floor(edges[1:]).astype(int), starts + 1)
    return np.stack([starts, ends], axis=1)


def segment_clip_indices(num_frames, n_segments=spec.N_SEGMENTS,
                         clip_len=spec.CLIP_LEN, stride=spec.FRAME_STRIDE):
    """(n_segments, clip_len) frame indices: one clip centred per segment."""
    bounds = segment_bounds(num_frames, n_segments)
    centers = (bounds[:, 0] + bounds[:, 1] - 1) / 2.0
    return np.stack([clip_indices(c, num_frames, clip_len, stride) for c in centers])


def clip_timestamps(t_end, clip_len=spec.CLIP_LEN, step_sec=spec.FRAME_STEP_SEC):
    """Target timestamps (seconds) of a live clip that ends at ``t_end``."""
    return t_end - (clip_len - 1 - np.arange(clip_len)) * step_sec
