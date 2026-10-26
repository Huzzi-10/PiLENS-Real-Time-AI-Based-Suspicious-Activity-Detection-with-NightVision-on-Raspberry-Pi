"""PiLENS: speed of CLIP ViT-B/16 (ONNX) on the Raspberry Pi 5, i.e. how long the alert confirmation takes.
On the Pi:  source venv/bin/activate ; pip install onnxruntime numpy
Run:        python clip_pi_bench.py --dir export_clip_b16

What it measures (on the Pi, nothing estimated):
  1. fp32 and int8 model: 1 frame and 4 frames (= the CLIP part of one clip), threads 2/3/4: median, p95
  2. Alert-window time: 1 clip (4 frames), 3 clips (12 frames), 5 clips (20 frames), with A0 in the same process
  3. Heat: runs for 3 minutes (--soak): temperature, MHz, throttle flag
Copy the output up to END OF REPORT.
"""
import os, time, json, argparse, subprocess
import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="export_clip_b16")
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--soak", type=float, default=3, help="minutes (0 = skip)")
a, _ = ap.parse_known_args()


def stats(v):
    v = np.asarray(v)
    return f"median {np.median(v):.0f} | p95 {np.percentile(v, 95):.0f} | mean {v.mean():.0f} +- {v.std():.0f} ms"


def temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception:
        return float("nan")


def freq():
    try:
        return int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read()) / 1000
    except Exception:
        return float("nan")


def throttled():
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
    except Exception:
        return "n/a"


def session(path, th):
    so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


models = [(n, os.path.join(a.dir, f)) for n, f in (("fp32", "clip_b16_fp32.onnx"), ("int8", "clip_b16_int8.onnx")) if os.path.exists(os.path.join(a.dir, f))]
assert models, f"{a.dir} mein clip_b16_*.onnx nahi mila"
print("SCRIPT VERSION: kd_pi_bench v1.0 | onnxruntime", ort.__version__, "| cpu temp start", temp(), "C | freq", freq(), "MHz | throttled", throttled())
for n, p in models:
    print(f"model {n}: {os.path.getsize(p) / 1e6:.0f} MB")

best, bestth = {}, {}
print("\n== 1. LATENCY (synthetic input 224x224) ==")
for n, p in models:
    for th in (2, 3, 4):
        s = session(p, th)
        for nf in (1, 4):
            x = np.random.randn(nf, 3, 224, 224).astype(np.float32)
            for _ in range(2):
                s.run(None, {"x": x})
            t = []
            for _ in range(a.runs):
                t0 = time.perf_counter(); s.run(None, {"x": x}); t.append((time.perf_counter() - t0) * 1000)
            print(f"{n} | threads {th} | {nf} frame(s): {stats(t)}")
            if nf == 4:
                best[(n, th)] = float(np.median(t))
for n, _ in models:
    th = min((t for (m, t) in best if m == n), key=lambda t: best[(n, t)])
    bestth[n] = th
    print(f"best {n}: {th} threads, 4 frames = {best[(n, th)]:.0f} ms | temp {temp()} C | throttled {throttled()}")

print("\n== 2. ALERT-WINDOW (A0 alert ke baad CLIP confirm), best threads par ==")
print("window | frames | " + " | ".join(f"{n} ms" for n, _ in models))
for clips in (1, 3, 5):
    row = []
    for n, p in models:
        s = session(p, bestth[n])
        x = np.random.randn(4 * clips, 3, 224, 224).astype(np.float32)
        s.run(None, {"x": x})
        t = []
        for _ in range(3):
            t0 = time.perf_counter(); s.run(None, {"x": x}); t.append((time.perf_counter() - t0) * 1000)
        row.append(f"{np.median(t):.0f}")
    print(f"{clips} clip(s) | {4 * clips} | " + " | ".join(row))

if a.soak > 0:
    n, p = models[0]
    th = bestth[n]
    print(f"\n== 3. SOAK {a.soak} min ({n}, {th} threads, 4 frames lagatar) ==")
    print("min | ms (median of window) | temp C | MHz | throttled")
    s = session(p, th); x = np.random.randn(4, 3, 224, 224).astype(np.float32)
    t0 = time.time(); win = []; last = t0
    while time.time() - t0 < a.soak * 60:
        t = time.perf_counter(); s.run(None, {"x": x}); win.append((time.perf_counter() - t) * 1000)
        if time.time() - last >= 30:
            print(f"{(time.time() - t0) / 60:4.1f} | {np.median(win):7.0f} | {temp():5.1f} | {freq():5.0f} | {throttled()}")
            win, last = [], time.time()
print("\nEND OF REPORT")
