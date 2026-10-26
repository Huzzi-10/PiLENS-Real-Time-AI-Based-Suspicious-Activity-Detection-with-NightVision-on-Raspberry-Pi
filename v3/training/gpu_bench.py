"""PiLENS GPU benchmark (Kaggle, one cell): the Pi speed numbers repeated on a GPU, as a reference.
Kaggle: Accelerator = GPU T4 x2 (or P100), Internet = ON, add the "pilens-gpu-bench" dataset (the ONNX files) as input.
Needs: pip install onnxruntime-gpu==1.23.2   (1.31 targets CUDA 13 and fails on Kaggle's CUDA 12.8)
Run (one cell):   !python /kaggle/input/pilens-gpu-bench/gpu_bench.py
What it measures (everything measured, nothing estimated):
  1. Every model (MoViNet-A0, A1, X3D-S backbone + heads, CLIP ViT-B/16): GPU latency at batch 1 (median / p95)
     - "GPU compute" (data already on the GPU) and "with copy" (from numpy, as in a camera pipeline)
  2. Batch throughput: batch 1/2/4/8/16/32 (clips per second) -> how many cameras one GPU can serve
  3. Concurrency: N threads on one session (N = 1/2/4/8) -> clips/s and p95
  4. CPU preprocessing per clip (Kaggle CPU) -> full clip latency
  5. GPU memory, 60 s sustained run (utilization %, temperature, power)
  6. Optional fp16 version (only if conversion and accuracy check pass)
Make sure the CLIP file is the real ViT-B/16 export (about 345 MB); a smaller file gives invalid CLIP numbers.
Output: /kaggle/working/gpu_bench_report.md + gpu_bench_results.json.
Flags: --root DIR (where the models are), --runs N, --soak SEC, --allow-cpu (testing only).
"""
import os, sys, time, json, glob, argparse, subprocess, threading

VERSION = "gpu_bench v1.2"
print("SCRIPT VERSION:", VERSION)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/kaggle/input" if os.path.isdir("/kaggle/input") else ".")
ap.add_argument("--out", default="/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
ap.add_argument("--runs", type=int, default=100)
ap.add_argument("--soak", type=float, default=60, help="seconds")
ap.add_argument("--allow-cpu", action="store_true", help="GPU na ho to CPU par chalao (sirf script test)")
ap.add_argument("--no-fp16", action="store_true")
ap.add_argument("--quick", action="store_true", help="test: kam batches/threads, chhote runs")
a, _ = ap.parse_known_args()


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


# ---- onnxruntime-gpu tayyar karo (Kaggle par CUDA 12 hai, isliye CUDA-12 wala version pin) ----
ORT_PIN = "onnxruntime-gpu==1.23.2"
SELFTEST = r"""
try:
    import torch
except Exception:
    pass
import onnx, onnxruntime as ort
from onnx import helper, TensorProto
try:
    ort.preload_dlls()
except Exception:
    pass
n = helper.make_node("Identity", ["x"], ["y"])
g = helper.make_graph([n], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 9
s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("SELFTEST", "OK" if "CUDAExecutionProvider" in s.get_providers() else "NO")
"""


def cuda_works():
    r = subprocess.run([sys.executable, "-c", SELFTEST], capture_output=True, text=True)
    return "SELFTEST OK" in r.stdout


if not a.allow_cpu and not cuda_works():
    if os.environ.get("PILENS_REEXEC"):
        print("FAIL: CUDA provider ab bhi nahi chala. Accelerator GPU hai? torch cuda:", sh("python -c 'import torch;print(torch.version.cuda)'"))
        print(subprocess.run([sys.executable, "-c", SELFTEST], capture_output=True, text=True).stderr[-1500:])
        sys.exit(1)
    print(f"CUDA onnxruntime nahi chala, {ORT_PIN} install ho raha hai (1-2 min)...")
    os.system(f"{sys.executable} -m pip uninstall -y onnxruntime onnxruntime-gpu -q")
    os.system(f"{sys.executable} -m pip install -q {ORT_PIN}")
    os.environ["PILENS_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

try:
    import torch  # CUDA libraries load karne ke liye (Kaggle par bundled)
    print("torch", torch.__version__, "cuda", torch.version.cuda)
except Exception:
    pass
import numpy as np
import onnx
import onnxruntime as ort
try:
    ort.preload_dlls()
except Exception:
    pass

PROV = ["CUDAExecutionProvider", "CPUExecutionProvider"]
GPU = "CUDAExecutionProvider" in ort.get_available_providers()
if not GPU and not a.allow_cpu:
    print("FAIL: CUDAExecutionProvider nahi mila. Accelerator GPU hai? Internet ON hai? (providers:", ort.get_available_providers(), ")")
    sys.exit(1)
if not GPU:
    print("WARNING: GPU nahi, CPU par chal raha hai (test mode). Ye numbers GPU ke nahi hain.")
print("onnxruntime", ort.__version__, "| providers", ort.get_available_providers())
print("GPU:", sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader") or "none")
print("CPU:", os.cpu_count(), "cores")
RESULTS = {"version": VERSION, "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"), "ort": ort.__version__, "models": {}}
REPORT = []


def out(s=""):
    print(s); REPORT.append(s)


def gpu_stats():
    # sirf GPU 0 (Kaggle T4 x2 par 2 lines aati hain; v1.1 mein isi wajah se sab 0 tha)
    r = sh("nvidia-smi -i 0 --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits")
    vals = (r.splitlines() or [""])[0].split(",")
    def f(i):
        try:
            return float(vals[i])
        except Exception:
            return float("nan")
    return dict(util=f(0), mem=f(1), temp=f(2), power=f(3), clk=f(4))


def pct(v, q):
    return float(np.percentile(v, q))


def sess(path, providers=None):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=providers or (PROV if GPU else ["CPUExecutionProvider"]))


def dyn_batch(src, dst):
    """Batch axis dynamic banao (ONNX ki input/output/value_info ki pehli dim)."""
    m = onnx.load(src)
    for t in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        d = t.type.tensor_type.shape.dim
        if len(d) > 0 and (d[0].dim_value == 1 or d[0].dim_param):
            d[0].dim_param = "N"
    onnx.save(m, dst)
    return dst


def time_gpu(s, feeds, runs, warm=10):
    """Data pehle se GPU par (OrtValue), sirf GPU compute. Har run ke baad synchronize."""
    if not GPU:
        return time_np(s, feeds, runs, warm)
    io = s.io_binding()
    for k, v in feeds.items():
        io.bind_ortvalue_input(k, ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0))
    for o in s.get_outputs():
        io.bind_output(o.name, "cuda", 0)
    for _ in range(warm):
        s.run_with_iobinding(io); io.synchronize_outputs()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); s.run_with_iobinding(io); io.synchronize_outputs(); t.append((time.perf_counter() - t0) * 1000)
    return t


def time_np(s, feeds, runs, warm=10):
    """numpy input/output: CPU->GPU copy + compute + GPU->CPU copy (camera pipeline jaisa)."""
    for _ in range(warm):
        s.run(None, feeds)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); s.run(None, feeds); t.append((time.perf_counter() - t0) * 1000)
    return t


def row(name, t):
    return f"{name}: median {np.median(t):.2f} | p95 {pct(t, 95):.2f} | mean {np.mean(t):.2f} +- {np.std(t):.2f} ms"


# ---- models dhoondo ----
metas = sorted(glob.glob(os.path.join(a.root, "**", "heads_meta.json"), recursive=True))
clips = sorted(glob.glob(os.path.join(a.root, "**", "clip_b16_fp32.onnx"), recursive=True))
print("\nmilay:", [os.path.dirname(m) for m in metas], clips)
assert metas or clips, f"{a.root} mein koi model nahi mila"

HOP_S = 0.6   # ek camera ko har 0.6 s mein ek naya clip chahiye (30 FPS, hop 18 frames)
SUMMARY = []

for mp in metas:
    d = os.path.dirname(mp)
    meta = json.load(open(mp))
    bbn = meta.get("backbone") or "backbone_x3ds.onnx"
    bb = os.path.join(d, bbn)
    hd = os.path.join(d, "heads.onnx")
    if not (os.path.exists(bb) and os.path.exists(hd)):
        print("skip (files kam):", d); continue
    name = meta.get("model", os.path.basename(d))
    cl = meta.get("clip", {}); NF, CROP = cl.get("frames", 13), cl.get("crop", 160)
    R = RESULTS["models"][name] = {}
    out(f"\n==================== {name} (clip {NF} x {CROP}x{CROP}) ====================")
    m0 = gpu_stats()["mem"]
    s1 = sess(bb)
    sh_ = sess(hd)
    x1 = np.random.rand(1, 3, NF, CROP, CROP).astype(np.float32)
    feat = np.random.randn(1, 2048).astype(np.float32)
    recent = np.random.randn(1, meta.get("recent_clips", 5), 2048).astype(np.float32)
    s1.run(None, {"clip": x1})
    R["gpu_mem_mb_after_load"] = gpu_stats()["mem"] - m0
    out(f"model sizes MB: backbone {os.path.getsize(bb) / 1e6:.1f}, heads {os.path.getsize(hd) / 1e6:.1f} | GPU memory used by this model (after 1 run): {R['gpu_mem_mb_after_load']:.0f} MB")

    out("\n-- 1. Latency, batch 1 --")
    tg = time_gpu(s1, {"clip": x1}, a.runs); tn = time_np(s1, {"clip": x1}, a.runs)
    th = time_np(sh_, {"feat": feat, "recent": recent}, a.runs)
    out(row("backbone, GPU compute only ", tg)); out(row("backbone, with CPU<->GPU copy", tn)); out(row("heads (with copy)            ", th))
    R["b1_gpu_ms"] = float(np.median(tg)); R["b1_gpu_p95"] = pct(tg, 95); R["b1_copy_ms"] = float(np.median(tn)); R["b1_copy_p95"] = pct(tn, 95)
    R["heads_ms"] = float(np.median(th))

    # CPU preprocessing (Kaggle CPU, resize+crop+normalise, NF frames 480x640)
    import cv2
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(NF)]
    SIDE = cl.get("side", 196)

    def pre():
        o = []
        for fr in frames:
            h, w = fr.shape[:2]; s = SIDE / min(h, w)
            fr2 = cv2.resize(fr, (int(round(w * s)), int(round(h * s)))); h, w = fr2.shape[:2]
            y0, x0 = (h - CROP) // 2, (w - CROP) // 2
            o.append(cv2.cvtColor(fr2[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_BGR2RGB))
        return np.ascontiguousarray(((np.stack(o).astype(np.float32) / 255 - cl.get("mean", 0)) / cl.get("std", 1)).transpose(3, 0, 1, 2)[None])
    tp = []
    for _ in range(30):
        t0 = time.perf_counter(); pre(); tp.append((time.perf_counter() - t0) * 1000)
    R["preprocess_ms"] = float(np.median(tp))
    tot = R["preprocess_ms"] + R["b1_copy_ms"] + R["heads_ms"]
    R["clip_total_ms"] = tot
    out(row("preprocess per clip (Kaggle CPU)", tp))
    out(f"CLIP LATENCY (preprocess + backbone with copy + heads) = {tot:.1f} ms  (hop budget {HOP_S * 1000:.0f} ms -> margin {HOP_S * 1000 / tot:.1f}x)")

    if GPU:
        out("\n-- 1b. Kaun sa node GPU par, kaun sa CPU par (ORT profiling, 3 runs) --")
        try:
            so = ort.SessionOptions(); so.enable_profiling = True
            so.profile_file_prefix = f"/tmp/prof_{name}"
            sp = ort.InferenceSession(bb, so, providers=PROV)
            for _ in range(3):
                sp.run(None, {"clip": x1})
            pf = sp.end_profiling(); ev = json.load(open(pf))
            from collections import defaultdict
            cnt, dur, opt = defaultdict(int), defaultdict(float), defaultdict(float)
            for e in ev:
                if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
                    pr = e.get("args", {}).get("provider", "?"); op = e.get("args", {}).get("op_name", "?")
                    cnt[pr] += 1; dur[pr] += e.get("dur", 0) / 3000.0; opt[(pr, op)] += e.get("dur", 0) / 3000.0
            for pr in cnt:
                out(f"{pr}: {cnt[pr] // 3} nodes, {dur[pr]:.1f} ms per run")
            out("sab se mehngay 5 ops: " + ", ".join(f"{op}@{pr.replace('ExecutionProvider', '')} {t:.1f} ms" for (pr, op), t in sorted(opt.items(), key=lambda kv: -kv[1])[:5]))
            R["placement"] = {pr: dict(nodes=cnt[pr] // 3, ms=dur[pr]) for pr in cnt}
        except Exception as e:
            out(f"profiling skip: {str(e)[:150]}")
        out("\n-- 1c. TensorRT EP (fp16), agar Kaggle par TensorRT libraries hon --")
        try:
            os.makedirs("/tmp/trt", exist_ok=True)
            st = ort.InferenceSession(bb, providers=[("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": "/tmp/trt"}), "CUDAExecutionProvider"])
            if st.get_providers()[0] != "TensorrtExecutionProvider":
                out("TensorRT nahi chala (libraries nahi hain) -> skip. Paper mein: 'TensorRT not evaluated'.")
            else:
                t0 = time.time(); yt = st.run(None, {"clip": x1})[0]; out(f"engine build {time.time() - t0:.0f} s")
                yr = s1.run(None, {"clip": x1})[0]
                cos = float((yt * yr).sum() / (np.linalg.norm(yt) * np.linalg.norm(yr) + 1e-9))
                tt = time_np(st, {"clip": x1}, a.runs)
                out(f"cosine(TRT fp16, fp32) = {cos:.5f}"); out(row("TensorRT fp16 backbone, with copy", tt))
                R["trt_fp16"] = dict(cosine=cos, copy_ms=float(np.median(tt)))
        except Exception as e:
            out(f"TensorRT skip: {str(e)[:150]}")

    out("\n-- 2. Batch throughput (alag clips ek saath) --")
    ok_batch = False
    try:
        dyn = dyn_batch(bb, os.path.join("/tmp", f"dyn_{name}.onnx"))
        sd = sess(dyn)
        xa = np.random.rand(4, 3, NF, CROP, CROP).astype(np.float32)
        ya = sd.run(None, {"clip": xa})[0]
        yr = np.concatenate([s1.run(None, {"clip": xa[i:i + 1]})[0] for i in range(4)])
        diff = float(np.abs(ya - yr).max()); ok_batch = diff < 1e-3
        out(f"batch model check: max diff vs batch-1 = {diff:.2e} -> {'OK' if ok_batch else 'FAIL (batch numbers skip)'}")
    except Exception as e:
        out(f"batch model nahi bana: {str(e)[:150]}")
    R["batch"] = {}
    best_cps = 1000.0 / R["b1_gpu_ms"]
    if ok_batch:
        out("batch | ms per batch (median) | ms per clip | clips/s | cameras (1 clip/0.6 s each)")
        for b in ((1, 2) if a.quick else (1, 2, 4, 8, 16, 32)):
            try:
                xb = np.random.rand(b, 3, NF, CROP, CROP).astype(np.float32)
                t = time_gpu(sd, {"clip": xb}, max(20, a.runs // 3))
                med = float(np.median(t)); cps = b * 1000 / med
                R["batch"][b] = dict(ms=med, p95=pct(t, 95), clips_per_s=cps, cameras=cps * HOP_S)
                best_cps = max(best_cps, cps)
                out(f"{b:5d} | {med:8.2f} | {med / b:7.2f} | {cps:7.1f} | {cps * HOP_S:6.0f}")
            except Exception as e:
                out(f"{b:5d} | fail: {str(e)[:80]}"); break

    out("\n-- 3. Concurrency (N threads, ek session, batch 1, 6 s har) --")
    out("threads | clips/s | per-call median ms | p95 ms | cameras")
    R["threads"] = {}
    for n in ((1, 2) if a.quick else (1, 2, 4, 8)):
        DUR = 2 if a.quick else 6
        lat, stop = [], time.time() + DUR

        def worker():
            local = []
            while time.time() < stop:
                t0 = time.perf_counter(); s1.run(None, {"clip": x1}); local.append((time.perf_counter() - t0) * 1000)
            lat.extend(local)
        ths = [threading.Thread(target=worker) for _ in range(n)]
        [t.start() for t in ths]; [t.join() for t in ths]
        cps = len(lat) / DUR
        R["threads"][n] = dict(clips_per_s=cps, median=float(np.median(lat)), p95=pct(lat, 95), cameras=cps * HOP_S)
        out(f"{n:7d} | {cps:7.1f} | {np.median(lat):8.2f} | {pct(lat, 95):7.2f} | {cps * HOP_S:6.0f}")
    R["max_clips_per_s"] = max([best_cps] + [v["clips_per_s"] for v in R["threads"].values()])
    R["max_cameras_no_gate"] = R["max_clips_per_s"] * HOP_S

    if not a.no_fp16 and GPU:
        out("\n-- 4. fp16 (converted, optional) --")
        try:
            os.system(f"{sys.executable} -m pip install -q onnxconverter-common")
            from onnxconverter_common import float16
            m16 = float16.convert_float_to_float16(onnx.load(bb), keep_io_types=True)
            p16 = os.path.join("/tmp", f"fp16_{name}.onnx"); onnx.save(m16, p16)
            s16 = sess(p16)
            y16 = s16.run(None, {"clip": x1})[0]; y32 = s1.run(None, {"clip": x1})[0]
            cos = float((y16 * y32).sum() / (np.linalg.norm(y16) * np.linalg.norm(y32) + 1e-9))
            t16 = time_gpu(s16, {"clip": x1}, a.runs); t16n = time_np(s16, {"clip": x1}, a.runs)
            out(f"cosine(fp16, fp32 feature) = {cos:.5f}")
            out(row("fp16 backbone, GPU compute", t16)); out(row("fp16 backbone, with copy ", t16n))
            R["fp16"] = dict(cosine=cos, gpu_ms=float(np.median(t16)), copy_ms=float(np.median(t16n)))
            if not np.isfinite(cos) or cos < 0.999:
                out("NOTE: fp16 FAIL (cosine < 0.999 ya NaN): is model ka fp16 use nahi karna, paper mein fp32 quote karo.")
        except Exception as e:
            out(f"fp16 skip: {str(e)[:150]}")

    SUMMARY.append((name, R))
    del s1, sh_

# ---- CLIP ----
for cp in clips[:1]:
    out(f"\n==================== CLIP ViT-B/16 fp32 (alert confirm) ====================")
    m0 = gpu_stats()["mem"]
    sc = sess(cp)
    sc.run(None, {"x": np.random.rand(4, 3, 224, 224).astype(np.float32)})
    R = RESULTS["models"]["clip_b16_fp32"] = {"gpu_mem_mb_after_load": gpu_stats()["mem"] - m0}
    out(f"GPU memory: {R['gpu_mem_mb_after_load']:.0f} MB")
    for n in ((1, 4) if a.quick else (1, 4, 12, 32)):
        x = np.random.rand(n, 3, 224, 224).astype(np.float32)
        tg = time_gpu(sc, {"x": x}, max(20, a.runs // 3)); tn = time_np(sc, {"x": x}, max(20, a.runs // 3))
        R[f"frames_{n}"] = dict(gpu=float(np.median(tg)), copy=float(np.median(tn)), p95=pct(tn, 95))
        out(f"{n:3d} frames: GPU compute {np.median(tg):7.2f} ms | with copy {np.median(tn):7.2f} ms (p95 {pct(tn, 95):.2f})" + ("   <- ek alert confirm (4 frames)" if n == 4 else ""))
    SUMMARY.append(("clip_b16_fp32", R))

# ---- sustained (sab se pehla A0 jaisa model) ----
if a.soak > 0 and metas:
    name0 = next((n for n, _ in SUMMARY if "movinet_a0" in n), SUMMARY[0][0])
    mp0 = next(m for m in metas if json.load(open(m)).get("model") == name0)
    d0 = os.path.dirname(mp0); meta0 = json.load(open(mp0)); cl = meta0.get("clip", {})
    s0 = sess(os.path.join(d0, meta0.get("backbone")))
    x = np.random.rand(1, 3, cl.get("frames", 13), cl.get("crop", 160), cl.get("crop", 160)).astype(np.float32)
    out(f"\n==================== SUSTAINED {a.soak:.0f} s, {name0}, batch 1 lagatar ====================")
    out("sec | clip ms (median) | GPU util % | GPU mem MB | temp C | power W | SM clock MHz")
    t0 = time.time(); win = []; last = t0; RESULTS["soak"] = []
    while time.time() - t0 < a.soak:
        t = time.perf_counter(); s0.run(None, {"clip": x}); win.append((time.perf_counter() - t) * 1000)
        if time.time() - last >= 10:
            g = gpu_stats(); RESULTS["soak"].append(dict(sec=time.time() - t0, ms=float(np.median(win)), **g))
            out(f"{time.time() - t0:3.0f} | {np.median(win):7.2f} | {g['util']:5.0f} | {g['mem']:6.0f} | {g['temp']:4.0f} | {g['power']:5.0f} | {g['clk']:5.0f}")
            win, last = [], time.time()

# ---- Pi se comparison table ----
PI = {"movinet_a0_v2": 137, "movinet_a1_v2": 256, "x3d_s": 196}
out("\n==================== PI vs GPU (backbone, batch 1) ====================")
out("model | Pi 5 CPU ms | GPU ms (with copy) | speedup | max cameras on this GPU (no motion gate)")
for n, R in SUMMARY:
    if "b1_copy_ms" in R:
        pi = next((v for k, v in PI.items() if k in n or n in k), None)
        out(f"{n} | {pi if pi else '-'} | {R['b1_copy_ms']:.1f} | {(str(round(pi / R['b1_copy_ms'], 1)) + 'x') if pi else '-'} | {R['max_cameras_no_gate']:.0f}")
RESULTS["pi_ms"] = PI
json.dump(RESULTS, open(os.path.join(a.out, "gpu_bench_results.json"), "w"), indent=1, default=float)
open(os.path.join(a.out, "gpu_bench_report.md"), "w").write("\n".join(REPORT))
print(f"\nsaved: {a.out}/gpu_bench_report.md, gpu_bench_results.json")
print("\nEND OF REPORT")
