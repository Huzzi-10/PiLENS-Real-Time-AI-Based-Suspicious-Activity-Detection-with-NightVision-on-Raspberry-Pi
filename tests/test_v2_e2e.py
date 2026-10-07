# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""End-to-end smoke test on synthetic data: features -> both trainers ->
ONNX export (tiny backbone) -> offline replay -> alert eval -> live pipeline.
Needs torch + onnx + onnxruntime (skipped otherwise)."""

import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from pilens_v2 import spec

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def cli(*args, timeout=600):
    r = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def make_features(root):
    """Separable synthetic features: anomaly videos have a few 'event' clips."""
    rng = np.random.default_rng(0)
    feats, splits = root / "feats", root / "splits"
    feats.mkdir()
    splits.mkdir()
    rows, ids = [], {}
    proto = rng.normal(0, 1, (len(spec.CLASSES_14), spec.FEATURE_DIM))
    for c_i, c in enumerate(spec.CLASSES_14):
        for i in range(12):
            vid = f"{c}{i:03d}_x264" if c != "Normal" else f"Normal_Videos_{i:03d}_x264"
            x = rng.normal(0, 1, (spec.N_SEGMENTS, spec.FEATURE_DIM))
            if c != "Normal":
                x[10:14] += 1.5 * proto[c_i]
            x += 0.5 * proto[c_i]
            np.save(feats / f"{vid}.npy", x.astype(np.float16))
            rows.append({"video_id": vid, "label": c, "num_frames": 3000, "rel_path": vid + ".mp4"})
            part = "test" if i >= 9 else ("val" if i >= 7 else "train")
            ids.setdefault(part, []).append(vid)
    with open(feats / "video_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for part, v in ids.items():
        (splits / f"cls14_fold1_{part}.txt").write_text("\n".join(v) + "\n")
        (splits / f"anomaly_{part}.txt").write_text("\n".join(v) + "\n")
    ann = root / "temporal.txt"
    ann.write_text("".join(f"{v}.mp4 X {10 * 3000 // 32} {14 * 3000 // 32} -1 -1\n"
                           for v in ids["test"] if not v.startswith("Normal")))
    return feats, splits, ann


def make_video(path, seconds=8, fps=30):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(int(seconds * fps)):
        f = np.full((240, 320, 3), 40, np.uint8)
        if i > fps * 3:  # something moves after 3 s
            x = (i * 4) % 260
            f[60:200, x:x + 50] = 210
        w.write(f)
    w.release()


def test_pipeline_end_to_end(tmp_path):
    feats, splits, ann = make_features(tmp_path)
    runs = tmp_path / "runs"

    out = cli("pilens_v2.train.train_mil", "--feats", str(feats), "--splits", str(splits),
              "--test-ann", str(ann), "--seeds", "0", "1", "--iters", "200", "--eval-every", "50",
              "--batch", "8", "--device", "cpu", "--out", str(runs / "mil"))
    mil = json.loads((runs / "mil" / "summary.json").read_text())
    assert mil["video_auc"][0] > 0.9, out
    assert mil["frame_auc"][0] > 0.8, out

    out = cli("pilens_v2.train.train_cls14", "--feats", str(feats), "--splits", str(splits),
              "--folds", "1", "--seeds", "0", "--epochs", "15", "--device", "cpu",
              "--out", str(runs / "cls"))
    cls = json.loads((runs / "cls" / "summary.json").read_text())
    assert cls["accuracy_mean_std"][0] > 0.5, out
    assert len(cls["confusion_matrix_first_seed_summed_over_folds"]) == len(spec.CLASSES_14)

    exports = tmp_path / "exports"
    video = tmp_path / "clip.mp4"
    make_video(video)
    out = cli("pilens_v2.export.export_onnx", "--out", str(exports), "--backbone", "tiny",
              "--binary-ckpt", str(runs / "mil" / "seed0.pt"),
              "--cls-ckpt", str(runs / "cls" / "fold1_seed0.pt"), "--parity-video", str(video))
    meta = json.loads((exports / "pilens_v2.json").read_text())
    assert meta["parity_ok"] and meta["binary_head"]["trained"], out
    assert meta["threshold"] == pytest.approx(mil["runs"][0]["threshold"])

    cli("pilens_v2.bench.benchmark", "--exports", str(exports), "--warmup", "2", "--runs", "5",
        "--out", str(tmp_path / "bench"))
    assert list((tmp_path / "bench").glob("*_run.json"))

    # offline replay + alert evaluation
    index = tmp_path / "index.csv"
    index.write_text("video_id,label,rel_path\nclip,Fighting,clip.mp4\n")
    ids = tmp_path / "ids.txt"
    ids.write_text("clip\n")
    cli("pilens_v2.runtime.replay", "--exports", str(exports), "--root", str(tmp_path),
        "--index", str(index), "--ids", str(ids), "--out", str(tmp_path / "replay"))
    lines = (tmp_path / "replay" / "clip.csv").read_text().splitlines()
    assert lines[0].startswith("# frames=240") and len(lines) > 5
    ev = tmp_path / "ev.csv"
    ev.write_text("video_id,start_frame,end_frame\nclip,90,239\n")
    out = cli("pilens_v2.eval.alert_eval", "--replay", str(tmp_path / "replay"), "--ann", str(ev),
              "--threshold", "0.0", "--out", str(tmp_path / "replay" / "alert_eval.json"))
    assert "2 of 3" in out

    # live threaded pipeline on the file (threshold 0 -> it must alert)
    logs = tmp_path / "logs" / "live"
    out = cli("pilens_v2.runtime.run", "--exports", str(exports), "--source", str(video),
              "--threshold", "0", "--no-email", "--no-stream", "--cooldown", "100",
              "--clip-dir", str(tmp_path / "Intruders"), "--log-dir", str(logs), "--duration", "20")
    summary = json.loads((logs / "summary.json").read_text())
    assert summary["hops_scored"] > 0 and summary["alerts"] >= 1, out
    assert summary["capture_fps"] > 20
    assert list((tmp_path / "Intruders").glob("*.mp4"))

    # paper tables from all of the above
    md_path = tmp_path / "results.md"
    cli("pilens_v2.eval.make_tables", "--runs", str(runs), "--bench", str(tmp_path / "bench"),
        "--logs", str(tmp_path / "logs"), "--alert", str(tmp_path / "replay" / "alert_eval.json"),
        "--out", str(md_path))
    md = md_path.read_text()
    assert "| cls | linear |" in md and "| mil | mlp |" in md
    assert "Confusion matrix" in md and "| live | gate |" in md and "| 2 of 3 | on |" in md
    assert "_No results yet._" not in md
