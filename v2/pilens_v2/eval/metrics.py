# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Metrics for the paper. Pure numpy, so they also run on the Pi.

Binary:   video AUC, frame AUC, frame AP, precision/recall at a fixed threshold
14-class: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
Alerts:   k-of-n voting simulation, event-to-alert time, false alarms per hour
"""

import numpy as np

from ..preprocess import segment_bounds


# --- ranking metrics --------------------------------------------------------

def roc_auc(y_true, scores):
    """Area under ROC (Mann-Whitney U, ties get average rank)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true, scores):
    """AP = sum over thresholds of (R_k - R_{k-1}) * P_k (sklearn definition)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    # one point per distinct threshold
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


# --- segment scores -> frame scores ----------------------------------------

def segments_to_frames(seg_scores, num_frames):
    """Repeat each of the T segment scores over its frames (REAL frame count)."""
    seg_scores = np.asarray(seg_scores, dtype=np.float32)
    out = np.empty(int(num_frames), dtype=np.float32)
    for (s, e), v in zip(segment_bounds(num_frames, len(seg_scores)), seg_scores):
        out[s:e] = v
    return out


def gaussian_smooth(x, sigma):
    """1-D Gaussian smoothing over segments (sigma in segments, edge-reflect)."""
    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0 or len(x) < 2:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(x, r, mode="reflect") if len(x) > r else np.pad(x, r, mode="edge")
    return np.convolve(padded, k, mode="valid")


def binary_report(seg_scores, video_labels, frame_gt=None, num_frames=None, smooth_sigma=0.0):
    """seg_scores: {vid: (T,)}, video_labels: {vid: 0/1},
    frame_gt: {vid: (num_frames,) 0/1} for the test videos, num_frames: {vid: int}."""
    vids = sorted(seg_scores)
    rep = {"video_auc": roc_auc([video_labels[v] for v in vids],
                                [float(np.max(seg_scores[v])) for v in vids])}
    if frame_gt is not None:
        ys, ss = [], []
        for v in vids:
            s = gaussian_smooth(seg_scores[v], smooth_sigma)
            ss.append(segments_to_frames(s, num_frames[v]))
            ys.append(frame_gt[v][:num_frames[v]])
        y, s = np.concatenate(ys), np.concatenate(ss)
        rep["frame_auc"] = roc_auc(y, s)
        rep["frame_ap"] = average_precision(y, s)
    return rep


def select_threshold(y_true, scores):
    """Threshold that maximises F1 (use on VALIDATION scores only)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    best_thr, best_f1 = 0.5, -1.0
    for t in np.unique(s):
        pred = s >= t
        tp = (pred & y).sum()
        f1 = 2 * tp / max(1, pred.sum() + y.sum())
        if f1 > best_f1:
            best_thr, best_f1 = float(t), float(f1)
    return best_thr, best_f1


def precision_recall_at(y_true, scores, thr):
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores) >= thr
    tp = (pred & y).sum()
    return {"threshold": float(thr), "precision": float(tp / max(1, pred.sum())),
            "recall": float(tp / max(1, y.sum()))}


# --- 14-class ---------------------------------------------------------------

def classification_report(y_true, y_pred, n_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(np.float64)
    prec = tp / np.maximum(1, cm.sum(axis=0))
    rec = tp / np.maximum(1, cm.sum(axis=1))
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(1e-12, prec + rec), 0.0)
    present = cm.sum(axis=1) > 0
    return {
        "accuracy": float(tp.sum() / max(1, cm.sum())),
        "macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
        "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def mean_std(values):
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)


# --- alert logic simulation -------------------------------------------------

def simulate_alerts(times, scores, thr, k=2, n=3, cooldown=0.0):
    """Times (s) at which the k-of-n rule fires; ``cooldown`` s between alerts."""
    alerts, window, last = [], [], -np.inf
    for t, s in zip(times, scores):
        window.append(s >= thr)
        if len(window) > n:
            window.pop(0)
        if sum(window) >= k and t - last >= cooldown:
            alerts.append(float(t))
            last = t
            window.clear()
    return alerts


def event_to_alert(alert_times, event_start, event_end=None):
    """Seconds from event start to the first alert during the event (None = missed)."""
    for t in alert_times:
        if t >= event_start and (event_end is None or t <= event_end):
            return float(t - event_start)
    return None


def false_alarms_per_hour(alert_times, events, duration_sec, tolerance=0.0):
    """Alerts that fall outside every (start, end + tolerance) event window, per hour."""
    false = sum(1 for t in alert_times
                if not any(s <= t <= e + tolerance for s, e in events))
    return false / max(1e-9, duration_sec / 3600.0), false
