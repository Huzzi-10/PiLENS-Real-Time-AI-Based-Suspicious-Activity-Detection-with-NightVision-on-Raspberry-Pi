# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Event annotations -> frame-level ground truth.

Two sources:
  * the official ``Temporal_Anomaly_Annotation_for_Testing_Videos.txt``
    (``Arson011_x264.mp4  Arson  150  420  680  1267``, -1 = no event);
  * our manual xlsx, exported to CSV with columns
    ``video_id,start_frame,end_frame[,day_night,usable]`` (one row per event).

Frame GT is always built from the video's REAL frame count (the old
512-frame alignment bug is why the 0.7472 number must not be cited).
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .splits import video_id_of


def read_official_temporal(path):
    """video_id -> list of (start, end) frame intervals (inclusive)."""
    events = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        vid = video_id_of(parts[0])
        nums = [int(x) for x in parts[2:]]
        intervals = [(s, e) for s, e in zip(nums[0::2], nums[1::2]) if s >= 0 and e >= 0]
        events[vid] = intervals
    return events


def read_events_csv(path):
    """Manual annotation CSV -> (events, meta).

    events: video_id -> [(start, end), ...] (rows with usable == 0/false/no dropped)
    meta:   video_id -> {"day_night": ..., "usable": bool}
    """
    events, meta = defaultdict(list), {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = video_id_of(row["video_id"])
            usable = str(row.get("usable", "1")).strip().lower() not in {"0", "false", "no", "n"}
            meta[vid] = {"day_night": (row.get("day_night") or "").strip().capitalize(), "usable": usable}
            if not usable:
                continue
            s, e = row.get("start_frame", ""), row.get("end_frame", "")
            if str(s).strip() not in {"", "-1"} and str(e).strip() not in {"", "-1"}:
                events[vid].append((int(float(s)), int(float(e))))
    return dict(events), meta


def frame_labels(intervals, num_frames):
    """0/1 per frame; interval ends are inclusive and clamped to the video."""
    y = np.zeros(int(num_frames), dtype=np.uint8)
    for s, e in intervals:
        s, e = max(0, int(s)), min(int(num_frames) - 1, int(e))
        if e >= s:
            y[s:e + 1] = 1
    return y
