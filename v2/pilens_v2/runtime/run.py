# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Step 6: live threaded pipeline on the Pi.

    python -m pilens_v2.runtime.run --exports exports                     # Pi camera
    python -m pilens_v2.runtime.run --exports exports --source test.mp4   # file, paced at its fps
    python -m pilens_v2.runtime.run ... --duration 1800 --log-dir logs/fan_30min

Threads (notes, section 7):
  capture   camera -> ring buffer, never blocks (nothing slow happens here)
  motion    MOG2 gate on the latest frame at ~10 Hz
  inference every --hop s: clip from ring -> backbone -> binary head -> k-of-n
            vote -> on alert: stage 2 (14-class on cached features)
  alert     LED + buzzer, save the last --preroll s, email, alert log
  stream    Flask MJPEG on STREAM_HOST:STREAM_PORT (Tailscale IP, not public)

Everything measured is written to --log-dir: hops.csv (per-stage ms, score,
motion), alerts.csv, summary.json (capture fps, clip latency, temperature).
"""

import argparse
import csv
import json
import queue
import threading
import time
from pathlib import Path

from .. import spec
from ..bench.benchmark import stats
from ..bench.sysinfo import HealthLogger, system_info
from .core import Cascade, FrameRing, MotionGate, Prediction, Voter
from .io import PiCamera, Signals, VideoSource, save_clip, send_email, setting, start_stream


class Pipeline:
    def __init__(self, args, source):
        self.args, self.source = args, source
        self.cascade = Cascade(args.exports, args.threads, args.backbone, args.threshold, args.min_conf)
        self.ring = FrameRing(max(spec.CLIP_SPAN_SEC, args.preroll + args.postroll) + 1.0, source.fps)
        self.motion = None if args.no_motion else MotionGate(min_area_frac=args.min_area, persist=args.persist)
        self.voter = Voter(args.k, args.n, args.cooldown)
        self.signals = Signals(int(setting("LED_PIN", "17")), int(setting("BUZZER_PIN", "18")))
        self.alert_q = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        self.hops, self.alerts = [], []
        self.frames_captured = 0
        self.status = {"score": 0.0, "motion": False, "label": ""}
        self.t0 = None

    def now(self):
        return time.monotonic() - self.t0

    # -- threads -------------------------------------------------------------
    def capture_loop(self):
        n = 0
        while not self.stop.is_set():
            ok, frame = self.source.read()
            if not ok:
                self.stop.set()
                break
            t = n / self.source.fps if getattr(self.source, "is_file", False) else self.now()
            self.ring.append(t, frame)
            n += 1
            self.frames_captured = n

    def motion_loop(self):
        last_t = None
        while not self.stop.is_set():
            t, frame = self.ring.latest()
            if frame is not None and t != last_t:
                self.motion.update(frame, t)
                last_t = t
            time.sleep(0.1)

    def inference_loop(self):
        hop = self.args.hop
        next_t = time.monotonic() + hop
        while not self.stop.is_set():
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_t = max(next_t + hop, time.monotonic())
            t_end, _ = self.ring.latest()
            if t_end is None or self.ring.span() < spec.CLIP_SPAN_SEC:
                continue  # first 2.4 s: buffer still filling
            gate = self.motion.active(t_end) if self.motion else True
            row = {"t": round(t_end, 3), "wall": round(self.now(), 3), "motion": int(gate),
                   "preprocess_ms": "", "backbone_ms": "", "head_ms": "", "score": "", "alert": 0}
            if gate:
                t0 = time.perf_counter()
                frames = self.ring.clip_at(t_end)
                if frames is None:
                    continue
                clip = self.cascade.preprocess(frames)
                row["preprocess_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
                score, ms_bb, ms_head = self.cascade.step(clip)
                row.update(backbone_ms=round(ms_bb, 2), head_ms=round(ms_head, 2), score=round(score, 4))
                suspicious = score >= self.cascade.threshold
            else:
                score, suspicious = 0.0, False
            self.status.update(score=score, motion=gate)
            if self.voter.push(suspicious, t_end):
                t_s2 = time.perf_counter()
                pred = Prediction("Suspicious") if self.args.no_stage2 else self.cascade.classify()
                event = {"t": t_end, "issued_wall": self.now(), "score": score, "label": pred.label,
                         "top3": pred.top3, "stage2_ms": (time.perf_counter() - t_s2) * 1e3}
                self.status["label"] = pred.label
                row["alert"] = 1
                try:
                    self.alert_q.put_nowait(event)
                except queue.Full:
                    event["dropped"] = True
                print(f"[ALERT] t={t_end:.1f}s {pred.label}  " +
                      ", ".join(f"{c} {p:.2f}" for c, p in pred.top3))
            self.hops.append(row)

    def alert_loop(self):
        a = self.args
        while not (self.stop.is_set() and self.alert_q.empty()):
            try:
                ev = self.alert_q.get(timeout=0.5)
            except queue.Empty:
                continue
            threading.Thread(target=self.signals.pulse, args=(a.buzz_sec,), daemon=True).start()
            if a.postroll > 0:
                time.sleep(a.postroll)
            frames = [(t, f) for t, f in self.ring.since(ev["t"] - a.preroll) if t <= ev["t"] + a.postroll]
            path = save_clip(frames, a.clip_dir, ev["label"])
            details = "Top-3: " + ", ".join(f"{c} ({p:.2f})" for c, p in ev["top3"]) + f"\nScore: {ev['score']:.3f}"
            try:
                mail = send_email(path, ev["label"], details) if not a.no_email else "disabled"
            except Exception as e:  # network down must not kill the pipeline
                mail = f"failed: {e}"
            ev.update(clip=path, email=mail, done_wall=self.now())
            self.alerts.append(ev)
            print(f"[ALERT] saved {path}  email {mail}")

    def jpeg(self):
        import cv2
        _, frame = self.ring.latest()
        if frame is None:
            return None
        frame = frame.copy()
        s = self.status
        color = (0, 0, 255) if s["score"] >= self.cascade.threshold else (0, 200, 0)
        cv2.putText(frame, f"score {s['score']:.2f}  motion {'on' if s['motion'] else 'off'}  {s['label']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    # -- run -----------------------------------------------------------------
    def run(self):
        a = self.args
        self.t0 = time.monotonic()
        health = HealthLogger(period=2.0)
        health.start()
        threads = [threading.Thread(target=f, name=f.__name__, daemon=True)
                   for f in (self.capture_loop, self.inference_loop, self.alert_loop)
                   + ((self.motion_loop,) if self.motion else ())]
        for th in threads:
            th.start()
        if not a.no_stream:
            print("stream:", start_stream(self.jpeg))
        try:
            while not self.stop.is_set() and (a.duration <= 0 or self.now() < a.duration):
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        self.stop.set()
        for th in threads:
            th.join(timeout=60 if th.name == "alert_loop" else 5)
        health.stop()
        self.signals.set(False)
        self.source.release()
        self.write_logs(health)

    def write_logs(self, health):
        a = self.args
        out = Path(a.log_dir)
        out.mkdir(parents=True, exist_ok=True)
        wall = self.now()
        scored = [h for h in self.hops if h["backbone_ms"] != ""]
        col = lambda k: [h[k] for h in scored]
        summary = {
            "duration_s": wall, "frames_captured": self.frames_captured,
            "capture_fps": self.frames_captured / max(1e-9, wall),
            "hop_s": a.hop, "hops": len(self.hops), "hops_scored": len(scored),
            "motion_gate_skip_frac": 1 - len(scored) / max(1, len(self.hops)),
            "alerts": len(self.alerts),
            "alerts_per_hour": len(self.alerts) / max(1e-9, wall / 3600),
            "threshold": self.cascade.threshold, "k_of_n": [a.k, a.n], "cooldown_s": a.cooldown,
            "latency": {k: stats(col(k)) for k in ("preprocess_ms", "backbone_ms", "head_ms")} if scored else {},
            "clip_latency": stats([h["preprocess_ms"] + h["backbone_ms"] + h["head_ms"] for h in scored]) if scored else {},
            "health": health.summary(), "system": system_info(a.threads), "args": vars(a),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        if self.hops:
            with open(out / "hops.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.hops[0]))
                w.writeheader()
                w.writerows(self.hops)
        with open(out / "alerts.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "issued_wall", "done_wall", "score", "label", "top3", "stage2_ms", "clip", "email"])
            for e in self.alerts:
                w.writerow([round(e["t"], 3), round(e["issued_wall"], 3), round(e.get("done_wall", 0), 3),
                            round(e["score"], 4), e["label"], json.dumps(e["top3"]),
                            round(e["stage2_ms"], 3), e.get("clip"), e.get("email")])
        print(f"\ncapture {summary['capture_fps']:.1f} fps   hops {len(self.hops)} "
              f"(scored {len(scored)})   alerts {len(self.alerts)}")
        if scored:
            c = summary["clip_latency"]
            print(f"clip latency median {c['median_ms']:.1f} ms  p95 {c['p95_ms']:.1f} ms   logs -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", default="exports")
    ap.add_argument("--backbone", default="backbone.onnx")
    ap.add_argument("--source", default="picamera", help="picamera | video path | webcam index")
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=3, help="ORT threads (leave a core for capture)")
    ap.add_argument("--hop", type=float, default=0.6, help="seconds between scored clips")
    ap.add_argument("--threshold", type=float, default=None, help="default: value chosen on validation")
    ap.add_argument("--k", type=int, default=spec.VOTE_K)
    ap.add_argument("--n", type=int, default=spec.VOTE_N)
    ap.add_argument("--cooldown", type=float, default=30.0)
    ap.add_argument("--min-conf", type=float, default=spec.STAGE2_MIN_CONF)
    ap.add_argument("--no-motion", action="store_true", help="disable the motion gate (ablation)")
    ap.add_argument("--no-stage2", action="store_true", help="binary only, skip the 14-class head (ablation)")
    ap.add_argument("--min-area", type=float, default=0.002, help="min blob area, fraction of frame")
    ap.add_argument("--persist", type=int, default=3, help="motion checks a blob must persist")
    ap.add_argument("--preroll", type=float, default=5.0)
    ap.add_argument("--postroll", type=float, default=0.0)
    ap.add_argument("--buzz-sec", type=float, default=2.0)
    ap.add_argument("--clip-dir", default="Intruders")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = until Ctrl+C)")
    ap.add_argument("--log-dir", default="logs/run")
    args = ap.parse_args()

    if args.source == "picamera":
        w, h = (int(v) for v in args.size.split("x"))
        source = PiCamera((w, h), args.fps)
    else:
        source = VideoSource(args.source, realtime=True)
    Pipeline(args, source).run()


if __name__ == "__main__":
    main()
