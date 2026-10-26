"""PiLENS CNN baseline Step 5: Raspberry Pi 5 benchmark for the 2D CNN (MobileNetV3-Large) + LSTM / Transformer / meanpool heads.

On the Pi (venv active, export_mnv3l folder copied from Kaggle):
  python c_step5_pi_bench.py --dir export_mnv3l --soak 10
  python c_step5_pi_bench.py --dir export_mnv3l --video Steal.mp4          (optional end to end)
What it measures (on the Pi, nothing estimated):
  1. CNN latency per frame (batch 1), threads 1/2/3/4; then batch 3 and batch 13 at the best thread count
  2. Latency of each temporal head (lstm, transformer, meanpool)
  3. Clip cost two ways: (a) recompute = all 13 frames of each clip through the CNN + head
                         (b) streaming = each frame's CNN feature computed once; per 0.6 s hop only 3 new frames + head
  4. (--video) end-to-end streaming loop on a real video: decode + preprocess + CNN + head
  5. (--soak N) N minutes of the recompute workload; every 30 s latency, CPU temperature, MHz, throttle flag
Copy the output up to END OF REPORT.
"""
import os, json, time, argparse, subprocess, collections
import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="export_mnv3l")
ap.add_argument("--video", default=None)
ap.add_argument("--runs", type=int, default=50)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--threads", default="1,2,3,4")
ap.add_argument("--soak", type=float, default=0, help="minutes")
ap.add_argument("--hop", type=int, default=18, help="naya clip har itne camera-frames par (18 = 0.6 sec @30FPS)")
ap.add_argument("--head", default="lstm", help="soak / video ke liye head: lstm, transformer ya meanpool")
a, _ = ap.parse_known_args()


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


def timeit(fn, runs=None):
    for _ in range(a.warmup):
        fn()
    out = []
    for _ in range(runs or a.runs):
        t = time.perf_counter(); fn(); out.append((time.perf_counter() - t) * 1000)
    return out


meta = json.load(open(os.path.join(a.dir, "heads_meta.json")))
CL = meta["clip"]
NF, STEP, SIDE, CROP, D = CL["frames"], CL["stride"], CL["side"], CL["crop"], meta["feat_dim"]
MEAN, STD = np.array(CL["mean"], np.float32), np.array(CL["std"], np.float32)
NEW = max(1, a.hop // STEP)                        # har hop par kitne naye sampled frames (18 / 6 = 3)
bbp = os.path.join(a.dir, meta["backbone"])
heads = {h: os.path.join(a.dir, v["file"]) for h, v in meta["heads"].items()}
print("model: mnv3l (MobileNetV3-Large per frame) + heads", list(heads), f"| clip {NF} x {CROP} | onnxruntime", ort.__version__,
      "| cpu temp start", temp(), "C | freq", freq(), "MHz | throttled", throttled())
print("model sizes MB: backbone", round(os.path.getsize(bbp) / 1e6, 1), "| " + " | ".join(f"{h} {os.path.getsize(p) / 1e6:.2f}" for h, p in heads.items()))

print("\n== 1. CNN LATENCY PER FRAME (synthetic input, batch 1) ==")
x1 = np.random.randn(1, 3, CROP, CROP).astype(np.float32)
best_th, best_med = 4, 1e9
for th in [int(t) for t in a.threads.split(",")]:
    sb = session(bbp, th)
    tb = timeit(lambda: sb.run(None, {"frames": x1}))
    print(f"threads {th}: {stats(tb)}")
    if np.median(tb) < best_med:
        best_th, best_med = th, float(np.median(tb))
print(f"best: {best_th} threads (per frame median {best_med:.1f} ms)")
sb = session(bbp, best_th)
x3, x13 = np.random.randn(NEW, 3, CROP, CROP).astype(np.float32), np.random.randn(NF, 3, CROP, CROP).astype(np.float32)
t3 = timeit(lambda: sb.run(None, {"frames": x3}), 30)
t13 = timeit(lambda: sb.run(None, {"frames": x13}), 20)
print(f"batch {NEW:2d} (streaming, ek hop ke naye frames): {stats(t3)}")
print(f"batch {NF:2d} (poora clip dobara)            : {stats(t13)}")

print(f"\n== 2. TEMPORAL HEADS ({best_th} threads, input 1 x {NF} x {D}) ==")
feats = np.random.randn(1, NF, D).astype(np.float32)
sh, th_med = {}, {}
for h, p in heads.items():
    sh[h] = session(p, best_th)
    th_ = timeit(lambda: sh[h].run(None, {"feats": feats}))
    th_med[h] = float(np.median(th_))
    print(f"{h:11s}: {stats(th_)} | params {meta['heads'][h]['params'] / 1e6:.3f} M")

print("\n== 3. CLIP COST (median, ms) vs hop budget", round(a.hop / 30 * 1000), "ms ==")
print("head        | recompute (13 frames + head) | streaming (", NEW, "new frames + head) | streaming % of hop")
for h in heads:
    rc, st = float(np.median(t13)) + th_med[h], float(np.median(t3)) + th_med[h]
    print(f"{h:11s} | {rc:28.1f} | {st:30.1f} | {100 * st / (a.hop / 30 * 1000):.0f}%")
print("(preprocess ka time --video wale section mein alag naapa jata hai)")


def prep(fr):
    import cv2
    h, w = fr.shape[:2]; s = SIDE / min(h, w)
    fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))))
    h, w = fr.shape[:2]; y0, x0 = (h - CROP) // 2, (w - CROP) // 2
    x = cv2.cvtColor(fr[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)


if a.video:
    import cv2
    hd = a.head if a.head in sh else next(iter(sh))
    print(f"\n== 4. END-TO-END STREAMING (decode + preprocess + CNN per sampled frame + {hd} head per hop) ==")
    cap = cv2.VideoCapture(a.video); buf = collections.deque(maxlen=NF); pend = []
    t_dec, t_pre, t_cnn, t_hd, t_hop, scores, n = [], [], [], [], [], [], 0
    while True:
        t = time.perf_counter(); ok, fr = cap.read(); t_dec.append((time.perf_counter() - t) * 1000)
        if not ok:
            break
        n += 1
        if n % STEP == 0:
            t = time.perf_counter(); pend.append(prep(fr)); t_pre.append((time.perf_counter() - t) * 1000)
        if n % a.hop == 0 and pend:
            t0 = time.perf_counter()
            f = sb.run(None, {"frames": np.stack(pend).astype(np.float32)})[0]; pend = []
            t_cnn.append((time.perf_counter() - t0) * 1000)
            buf.extend(list(f))
            if len(buf) == NF:
                t = time.perf_counter(); s = sh[hd].run(None, {"feats": np.stack(buf)[None].astype(np.float32)})[0]; t_hd.append((time.perf_counter() - t) * 1000)
                scores.append(float(np.ravel(s)[0])); t_hop.append((time.perf_counter() - t0) * 1000)
    if t_hop:
        print(f"frames {n}, clips scored {len(t_hop)}, hop {a.hop} frames")
        print("decode per frame        :", stats(t_dec)); print("preprocess per sampled fr:", stats(t_pre))
        print(f"CNN per hop ({NEW} frames)  :", stats(t_cnn)); print("head per hop            :", stats(t_hd))
        print("compute per hop (CNN+head):", stats(t_hop))
        thr = meta["heads"][hd]["thr_reco"]
        print(f"scores: max {max(scores):.3f} | clips above val-FPR5% thr {thr['val_fpr5']:.3f}: {sum(s >= thr['val_fpr5'] for s in scores)} of {len(scores)}")
    else:
        print(f"video bahut chhoti hai ({(NF - 1) * STEP + 1}+ frames chahiye)")

if a.soak > 0:
    hd = a.head if a.head in sh else next(iter(sh))
    print(f"\n== 5. SOAK {a.soak} min (recompute workload: 13 frames + {hd} head, lagatar) ==")
    print("min | clip ms (median of window) | temp C | MHz | throttled")
    t0 = time.time(); win = []; last = t0
    while time.time() - t0 < a.soak * 60:
        t = time.perf_counter(); f = sb.run(None, {"frames": x13})[0]; sh[hd].run(None, {"feats": f[None]}); win.append((time.perf_counter() - t) * 1000)
        if time.time() - last >= 30:
            print(f"{(time.time() - t0) / 60:4.1f} | {np.median(win):7.0f} | {temp():5.1f} | {freq():5.0f} | {throttled()}")
            win, last = [], time.time()
print("\nEND OF REPORT")
