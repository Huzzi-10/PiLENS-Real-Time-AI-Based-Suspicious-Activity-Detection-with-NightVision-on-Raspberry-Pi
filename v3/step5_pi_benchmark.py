"""PiLENS Step 5: benchmark on the Raspberry Pi 5 (ONNX Runtime, CPU).
On the Pi:  pip install onnxruntime numpy opencv-python   (inside the venv)
Run:        python step5_pi_benchmark.py --dir export_movinet --video test.mp4 --soak 10

What it measures (everything is measured on the Pi, nothing is estimated):
  1. Clip latency: backbone + heads with 1/2/3/4 threads (median, p95, mean, std)
  2. End to end (with --video): decode + preprocess + backbone + heads, each stage timed separately,
     and stream throughput (how many camera frames per second the pipeline can sustain)
  3. Soak (with --soak N): runs for N minutes; every 30 s logs latency, CPU temperature, clock and throttle flag
Copy the output up to END OF REPORT.
"""
import os, sys, time, json, argparse, subprocess
import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="export")
ap.add_argument("--video", default=None)
ap.add_argument("--runs", type=int, default=50)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--threads", default="1,2,3,4")
ap.add_argument("--soak", type=float, default=0, help="minutes")
ap.add_argument("--hop", type=int, default=18, help="naya clip har itne camera-frames par (18 = 0.6 sec @30FPS)")
a, _ = ap.parse_known_args()      # Jupyter/Colab ke extra arguments (-f ...) ignore


def stats(v):
    v = np.asarray(v)
    return f"median {np.median(v):.1f} | p95 {np.percentile(v, 95):.1f} | mean {v.mean():.1f} +- {v.std():.1f} ms"


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


meta = json.load(open(os.path.join(a.dir, "heads_meta.json")))
CL = meta.get("clip", {})
NF, STEP, SIDE, CROP = CL.get("frames", 13), CL.get("stride", 6), CL.get("side", 182), CL.get("crop", 160)
MEAN, STD = CL.get("mean", 0.45), CL.get("std", 0.225)
bbp, hp = os.path.join(a.dir, meta.get("backbone", "backbone_x3ds.onnx")), os.path.join(a.dir, "heads.onnx")
print("model:", meta.get("model", "x3d_s"), "| clip", NF, "x", CROP, "| onnxruntime", ort.__version__, "| cpu temp start", temp(), "C | freq", freq(), "MHz | throttled", throttled())
print("model sizes MB:", round(os.path.getsize(bbp) / 1e6, 1), round(os.path.getsize(hp) / 1e6, 1))
clip = np.random.randn(1, 3, NF, CROP, CROP).astype(np.float32)
feat, recent = np.random.randn(1, 2048).astype(np.float32), np.random.randn(1, meta["recent_clips"], 2048).astype(np.float32)

print("\n== 1. CLIP LATENCY (synthetic input) ==")
best_th, best_med = 4, 1e9
for th in [int(t) for t in a.threads.split(",")]:
    sb, sh = session(bbp, th), session(hp, th)
    for _ in range(a.warmup):
        sb.run(None, {"clip": clip}); sh.run(None, {"feat": feat, "recent": recent})
    tb, thh = [], []
    for _ in range(a.runs):
        t = time.perf_counter(); sb.run(None, {"clip": clip}); tb.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter(); sh.run(None, {"feat": feat, "recent": recent}); thh.append((time.perf_counter() - t) * 1000)
    print(f"threads {th}: backbone {stats(tb)}\n           heads    {stats(thh)}")
    if np.median(tb) < best_med:
        best_th, best_med = th, float(np.median(tb))
print(f"best: {best_th} threads (backbone median {best_med:.0f} ms) | cpu temp now {temp()} C | throttled {throttled()}")

sb, sh = session(bbp, best_th), session(hp, best_th)


def preprocess(frames):
    import cv2
    out = []
    for fr in frames:
        h, w = fr.shape[:2]; s = SIDE / min(h, w)
        fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))))
        h, w = fr.shape[:2]; y0, x0 = (h - CROP) // 2, (w - CROP) // 2
        out.append(cv2.cvtColor(fr[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_BGR2RGB))
    x = (np.stack(out).astype(np.float32) / 255 - MEAN) / STD           # T,H,W,C
    return np.ascontiguousarray(x.transpose(3, 0, 1, 2)[None])


if a.video:
    import cv2
    print("\n== 2. END-TO-END (decode + preprocess + backbone + heads), single thread loop ==")
    cap = cv2.VideoCapture(a.video); span = (NF - 1) * STEP + 1; buf = []
    t_dec, t_pre, t_bb, t_hd, nclip, nframes, hist = [], [], [], [], 0, 0, []
    t_all = time.perf_counter()
    while True:
        t = time.perf_counter(); ok, fr = cap.read(); t_dec.append((time.perf_counter() - t) * 1000)
        if not ok:
            break
        nframes += 1; buf.append(fr); buf = buf[-span:]
        if len(buf) == span and nframes % a.hop == 0:
            t = time.perf_counter(); x = preprocess(buf[::STEP]); t_pre.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); f = sb.run(None, {"clip": x})[0]; t_bb.append((time.perf_counter() - t) * 1000)
            hist = (hist + [f[0]])[-meta["recent_clips"]:]
            rec = np.stack((hist * meta["recent_clips"])[:meta["recent_clips"]])[None].astype(np.float32)
            t = time.perf_counter(); sh.run(None, {"feat": f, "recent": rec}); t_hd.append((time.perf_counter() - t) * 1000)
            nclip += 1
    tot = time.perf_counter() - t_all
    if nclip:
        print(f"frames {nframes}, clips scored {nclip}, hop {a.hop} frames")
        print("decode per frame :", stats(t_dec)); print("preprocess/clip  :", stats(t_pre))
        print("backbone/clip    :", stats(t_bb)); print("heads/clip       :", stats(t_hd))
        per_clip = np.median(t_pre) + np.median(t_bb) + np.median(t_hd)
        print(f"clip latency (preprocess+backbone+heads): {per_clip:.0f} ms")
        print(f"pipeline ne {nframes / tot:.1f} frames/sec process kiye (decode ke saath, file speed par) | max stream FPS jo hop={a.hop} ke saath real-time rahe: "
              f"{a.hop / (per_clip / 1000 + a.hop * np.median(t_dec) / 1000):.1f}")
    else:
        print(f"video bahut chhoti hai ({span}+ frames chahiye)")

if a.soak > 0:
    print(f"\n== 3. SOAK {a.soak} min ==")
    print("min | clip ms (median of window) | temp C | MHz | throttled")
    t0 = time.time(); win = []; last = t0
    while time.time() - t0 < a.soak * 60:
        t = time.perf_counter(); sb.run(None, {"clip": clip}); win.append((time.perf_counter() - t) * 1000)
        if time.time() - last >= 30:
            print(f"{(time.time() - t0) / 60:4.1f} | {np.median(win):7.0f} | {temp():5.1f} | {freq():5.0f} | {throttled()}")
            win, last = [], time.time()
print("\nEND OF REPORT")
