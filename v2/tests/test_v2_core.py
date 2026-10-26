# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Fast unit tests (numpy / OpenCV only): python -m pytest tests -q"""

import numpy as np
import pytest

from pilens_v2 import spec
from pilens_v2.data.annotations import frame_labels
from pilens_v2.data.splits import build, label_of, stratified_val, video_id_of
from pilens_v2.eval.metrics import (average_precision, classification_report, event_to_alert,
                                    false_alarms_per_hour, gaussian_smooth, roc_auc,
                                    segments_to_frames, simulate_alerts)
from pilens_v2.preprocess import (clip_indices, clip_timestamps, frames_to_clip, prepare_frame,
                                  segment_bounds, segment_clip_indices)
from pilens_v2.runtime.core import FrameRing, MotionGate, Voter


# --- preprocessing -------------------------------------------------------------

def test_clip_shape_and_normalisation():
    frames = [np.full((240, 320, 3), 115, np.uint8)] * spec.CLIP_LEN
    x = frames_to_clip(frames)
    assert x.shape == (3, spec.CLIP_LEN, spec.CROP_SIZE, spec.CROP_SIZE)
    assert x.dtype == np.float32
    assert np.allclose(x, (115 / 255 - 0.45) / 0.225, atol=1e-5)


def test_gray3_channels_equal_and_squash_keeps_edges():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    g = prepare_frame(f, gray3=True)
    assert np.array_equal(g[..., 0], g[..., 1]) and np.array_equal(g[..., 1], g[..., 2])
    edge = np.zeros((240, 320, 3), np.uint8)
    edge[:, :20] = 255  # event at the left edge
    assert prepare_frame(edge, crop_mode="center").max() == 0  # cut by the center crop
    assert prepare_frame(edge, crop_mode="squash").max() > 0


def test_segment_sampling():
    n = 9000
    bounds = segment_bounds(n)
    assert len(bounds) == spec.N_SEGMENTS and bounds[0, 0] == 0 and bounds[-1, 1] == n
    idx = segment_clip_indices(n)
    assert idx.shape == (spec.N_SEGMENTS, spec.CLIP_LEN)
    assert idx.min() >= 0 and idx.max() < n
    assert np.all(np.diff(idx[5]) == spec.FRAME_STRIDE)
    short = segment_clip_indices(20)  # shorter than one clip: clamped, no crash
    assert short.min() >= 0 and short.max() <= 19
    assert list(clip_indices(0, 100))[:7] == [0] * 7
    ts = clip_timestamps(10.0)
    assert ts[-1] == pytest.approx(10.0) and ts[0] == pytest.approx(10.0 - spec.CLIP_SPAN_SEC)


# --- splits --------------------------------------------------------------------

def test_names():
    assert video_id_of("Abuse/Abuse001_x264.mp4") == "Abuse001_x264"
    assert label_of("RoadAccidents012_x264") == "RoadAccidents"
    assert label_of("Normal_Videos_003_x264") == "Normal"
    assert label_of("Normal_Videos307_x264") == "Normal"
    assert label_of("Unknown123") is None


def _fake_dataset(root):
    classes = ["Abuse", "Fighting", "Robbery"]
    rng = np.random.default_rng(0)
    vids = []
    for c in classes:
        for i in range(1, 11):
            p = root / c / f"{c}{i:03d}_x264.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(rng.bytes(1000 + i))
            vids.append(f"{c}/{p.name}")
    normals = []
    for i in range(1, 11):
        data = rng.bytes(5000 + i)
        for folder in ("Normal_Videos_for_Event_Recognition", "Testing_Normal_Videos_Anomaly"):
            p = root / folder / f"Normal_Videos_{i:03d}_x264.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)  # byte-identical duplicate
        normals.append(f"Normal_Videos_event/Normal_Videos_{i:03d}_x264.mp4")
    # 14-class lists (one fold is enough for the test)
    sp = root / "Action_Regnition_splits"
    sp.mkdir()
    train = [v for v in vids + normals if int(v[-12:-9]) <= 8]
    test = [v for v in vids + normals if int(v[-12:-9]) > 8]
    (sp / "train_001.txt").write_text("\n".join(train) + "\n")
    (sp / "test_001.txt").write_text("\n".join(test) + "\n")
    # anomaly lists: the nested Anomaly_Train.txt is EMPTY, root one is real
    (root / "Anomaly_Detection_splits").mkdir()
    (root / "Anomaly_Detection_splits" / "Anomaly_Train.txt").write_text("")
    (root / "Anomaly_Train.txt").write_text("\n".join(train) + "\n")
    (root / "Anomaly_Test.txt").write_text("\n".join(test).replace("Normal_Videos_event", "Testing_Normal_Videos_Anomaly") + "\n")
    return len(vids) + len(normals)


def test_build_splits_dedupes_and_never_leaks(tmp_path):
    root, out = tmp_path / "data", tmp_path / "splits"
    n_unique = _fake_dataset(root)
    s = build(root, out)
    assert s["unique_videos"] == n_unique and s["duplicates_removed"] == 10
    a = s["tasks"]["anomaly"]
    assert "Anomaly_Detection_splits" in a["skipped_empty"][0]
    assert a["train"] + a["val"] == 32 and a["test"] == 8 and not a["missing_from_disk"]
    parts = {p: set((out / f"anomaly_{p}.txt").read_text().split()) for p in ("train", "val", "test")}
    assert not (parts["train"] & parts["val"]) and not (parts["train"] & parts["test"])
    f1 = s["tasks"]["cls14"]["fold1"]
    assert f1["test"] == 8 and f1["val"] >= 4  # every class in val
    assert (out / "split_summary.json").exists()


def test_stratified_val_is_deterministic():
    ids = [f"v{i}" for i in range(40)]
    labels = {v: ("A" if i % 2 else "B") for i, v in enumerate(ids)}
    assert stratified_val(ids, labels, 0.15, 3) == stratified_val(ids, labels, 0.15, 3)
    tr, va = stratified_val(ids, labels, 0.15, 3)
    assert len(va) == 6 and not set(tr) & set(va)


# --- metrics -------------------------------------------------------------------

def test_auc_ap_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500) + 0.3 * y, 2)  # rounding creates ties
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-9)
    assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-9)


def test_frame_expansion_uses_real_frame_count():
    f = segments_to_frames(np.arange(32), 1000)
    assert len(f) == 1000 and f[0] == 0 and f[-1] == 31
    assert np.all(np.diff(f) >= 0)
    y = frame_labels([(10, 19), (990, 2000)], 1000)
    assert y.sum() == 10 + 10
    sm = gaussian_smooth(np.r_[np.zeros(10), 1.0, np.zeros(10)], 1.0)
    assert len(sm) == 21 and sm[10] < 1.0 and sm.sum() == pytest.approx(1.0, abs=1e-6)


def test_classification_report():
    r = classification_report([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], 3)
    assert r["accuracy"] == pytest.approx(0.6)
    assert r["confusion_matrix"][0] == [1, 1, 0]
    assert r["f1"][2] == 0.0


def test_alert_simulation_and_voter_agree():
    t = np.arange(20) * 0.6
    s = np.zeros(20)
    s[[5, 7, 8, 15]] = 0.9
    alerts = simulate_alerts(t, s, 0.5, k=2, n=3, cooldown=0)
    assert alerts == [pytest.approx(t[7])]  # 5 and 7 within 3 hops; 15 alone does not fire
    v = Voter(2, 3, cooldown_sec=0)
    fired = [ti for ti, si in zip(t, s) if v.push(si >= 0.5, ti)]
    assert fired == pytest.approx(alerts)
    assert simulate_alerts(t, s, 0.5, k=1, n=1, cooldown=100) == [pytest.approx(t[5])]
    assert event_to_alert(alerts, 2.0, 6.0) == pytest.approx(t[7] - 2.0)
    assert event_to_alert(alerts, 6.0) is None
    fah, n_false = false_alarms_per_hour([1.0, 50.0], [(0.0, 2.0)], 3600)
    assert n_false == 1 and fah == pytest.approx(1.0)


# --- runtime pieces -----------------------------------------------------------------

def test_ring_clip_at():
    ring = FrameRing(seconds=3.0, fps=30)
    for i in range(120):
        ring.append(i / 30, np.full((4, 4, 3), i, np.uint8))
    clip = ring.clip_at(119 / 30)
    assert len(clip) == spec.CLIP_LEN
    assert [int(f[0, 0, 0]) for f in clip] == list(range(119 - 72, 120, 6))
    assert FrameRing(1.0, 30).clip_at(1.0) is None


def test_motion_gate_filters_static_and_one_frame_blobs():
    gate = MotionGate(persist=3, hold_sec=0.0)
    bg = np.full((180, 320, 3), 60, np.uint8)
    for i in range(60):
        assert not gate.update(bg, i / 10)
    blink = bg.copy()
    blink[50:80, 50:80] = 255  # one-frame flash (insect / flicker)
    assert not gate.update(blink, 6.0)
    for i in range(6):
        assert not gate.update(bg, 6.1 + i / 10)
    active = False
    for i in range(10):  # a person-sized block that keeps moving
        f = bg.copy()
        f[60:140, 20 + 15 * i:70 + 15 * i] = 220
        active = gate.update(f, 7.0 + i / 10)
    assert active and gate.box is not None


def test_daynight_calibration_separates_dark_gray_from_day():
    from pilens_v2.data.daynight import calibrate, predict
    rng = np.random.default_rng(0)
    day = [((rng.uniform(60, 120), rng.uniform(90, 160)), "Day") for _ in range(30)]
    night = [((rng.uniform(0, 15), rng.uniform(20, 140)), "Night") for _ in range(30)]
    cal = calibrate(day + night)
    assert cal["balanced_acc"] == 1.0
    assert predict((5.0, 120.0), cal["sat_thr"], cal["val_thr"]) == "Night"
    assert predict((90.0, 120.0), cal["sat_thr"], cal["val_thr"]) == "Day"
