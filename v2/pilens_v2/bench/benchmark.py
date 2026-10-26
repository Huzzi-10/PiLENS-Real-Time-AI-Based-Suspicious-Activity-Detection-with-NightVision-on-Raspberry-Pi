# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Step 5: FP32 (then INT8) clip-latency benchmark on the Raspberry Pi 5.

    # clip latency: 20 warm-up + 200 timed runs, per stage
    python -m pilens_v2.bench.benchmark --exports exports --threads 4 --tag fp32_fan
    # long run (30 min) for the temperature / throughput graph
    python -m pilens_v2.bench.benchmark --exports exports --duration 1800 --tag fp32_nofan_30min
    # INT8 backbone
    python -m pilens_v2.bench.benchmark --exports exports --backbone backbone_int8.onnx --tag int8

Stages timed separately: preprocess (13 raw frames -> tensor), backbone,
binary head, cls14 head. This is the MODEL-ONLY number; the full-system
numbers (capture + motion gate + alert thread running) come from the runtime's
per-hop log (python -m pilens_v2.runtime.run --log-dir ...).
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import spec
from ..preprocess import frames_to_clip
from .sysinfo import HealthLogger, system_info, throttled


def stats(ms):
    a = np.asarray(ms, dtype=np.float64)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min_ms": float(a.min()), "max_ms": float(a.max()), "n": int(len(a))}


def session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def load_frames(video, size):
    """13 raw frames: from a video if given, else synthetic noise at camera size."""
    if video:
        import cv2
        cap = cv2.VideoCapture(str(video))
        frames = []
        while len(frames) < spec.CLIP_LEN:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if len(frames) == spec.CLIP_LEN:
            return frames
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) for _ in range(spec.CLIP_LEN)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", default="exports")
    ap.add_argument("--backbone", default="backbone.onnx")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--duration", type=float, default=0, help="seconds; >0 = long thermal run")
    ap.add_argument("--video", default=None, help="take the 13 frames from this video")
    ap.add_argument("--frame-size", default="640x480", help="synthetic frame size WxH")
    ap.add_argument("--gray3", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="bench_results")
    args = ap.parse_args()

    ex = Path(args.exports)
    size = tuple(int(v) for v in args.frame_size.split("x"))
    frames = load_frames(args.video, size)
    sess_bb = session(ex / args.backbone, args.threads)
    sess_bin = session(ex / "binary_head.onnx", args.threads)
    sess_cls = session(ex / "cls14_head.onnx", args.threads)
    hist = np.zeros((1, spec.STAGE2_CLIPS, spec.FEATURE_DIM), np.float32)

    def one():
        t0 = time.perf_counter()
        clip = frames_to_clip(frames, args.gray3)[None]
        t1 = time.perf_counter()
        feat = sess_bb.run(None, {"clip": clip})[0]
        t2 = time.perf_counter()
        hist[0, :-1] = hist[0, 1:]
        hist[0, -1] = feat[0]
        sess_bin.run(None, {"feats": hist[:, -spec.VOTE_N:]})
        t3 = time.perf_counter()
        sess_cls.run(None, {"feats": hist})
        t4 = time.perf_counter()
        return [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    health = HealthLogger(period=1.0)
    health.start()

    for _ in range(args.warmup):
        one()
    rows, timeline = [], []
    t_start = time.time()
    if args.duration > 0:
        sec, done = 0, []
        while time.time() - t_start < args.duration:
            rows.append(one())
            done.append(rows[-1])
            if time.time() - t_start >= sec + 10:  # one timeline row per 10 s
                h = health.samples[-1] if health.samples else {}
                lat = [sum(r[:3]) for r in done]
                timeline.append({"t_s": round(time.time() - t_start, 1),
                                 "clips_per_s": len(done) / 10.0,
                                 "stage1_median_ms": float(np.median(lat)),
                                 "temp_c": h.get("temp_c"), "freq_mhz": h.get("freq_mhz")})
                print(timeline[-1])
                sec += 10
                done = []
    else:
        rows = [one() for _ in range(args.runs)]
    wall = time.time() - t_start
    health.stop()

    r = np.asarray(rows)
    result = {
        "tag": args.tag, "timestamp": stamp, "backbone": args.backbone, "threads": args.threads,
        "warmup": args.warmup, "runs": len(rows), "wall_s": wall,
        "frames": "video" if args.video else f"synthetic {args.frame_size}", "gray3": args.gray3,
        "latency": {"preprocess": stats(r[:, 0]), "backbone": stats(r[:, 1]),
                    "binary_head": stats(r[:, 2]), "cls14_head": stats(r[:, 3]),
                    "stage1_total": stats(r[:, :3].sum(1)), "with_stage2": stats(r.sum(1))},
        "clips_per_s_back_to_back": len(rows) / wall,
        "health": health.summary(), "throttled_after": throttled(), "system": system_info(args.threads),
        "note": "model-only; measured, not estimated",
    }
    (out / f"{stamp}_{args.tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out / f"{stamp}_{args.tag}_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["preprocess_ms", "backbone_ms", "binary_head_ms", "cls14_head_ms"])
        w.writerows(np.round(r, 3).tolist())
    with open(out / f"{stamp}_{args.tag}_health.csv", "w", newline="") as f:
        if health.samples:
            w = csv.DictWriter(f, fieldnames=list(health.samples[0]))
            w.writeheader()
            w.writerows(health.samples)
    if timeline:
        with open(out / f"{stamp}_{args.tag}_timeline.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(timeline[0]))
            w.writeheader()
            w.writerows(timeline)

    print(f"\n{args.tag}  ({len(rows)} runs, {args.threads} threads)")
    for k, v in result["latency"].items():
        print(f"  {k:14s} median {v['median_ms']:8.2f} ms   p95 {v['p95_ms']:8.2f}   "
              f"mean {v['mean_ms']:8.2f} +- {v['std_ms']:.2f}")
    t = result["health"].get("temp_c")
    if t:
        print(f"  temp           start {t['start']:.1f} C  max {t['max']:.1f} C")
    print(f"  saved {out}/{stamp}_{args.tag}.json")


if __name__ == "__main__":
    main()
