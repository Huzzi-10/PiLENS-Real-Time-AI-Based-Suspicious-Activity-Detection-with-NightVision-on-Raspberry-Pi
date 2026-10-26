"""PiLENS MoViNet Step 0 (Kaggle): checks that MoViNet-A0 (Kinetics-600) loads and exports to ONNX. Trains nothing.
Kaggle notebook: GPU on, Internet on, add the dataset `minmints/ufc-crime-full-dataset`, paste this file into one cell and run.
Checks:
  1. The MoViNet code downloads from GitHub and the pretrained weights load
  2. Feature wrapper: clip (1,3,13,172,172) -> feature (1,2048), and it matches the pretrained classifier
  3. Input normalization (the confidence check is only informative; the probe AUC in the diagnostics showed [0,1] input is correct)
  4. ONNX export + onnxruntime parity + Kaggle CPU latency (not the Pi number)
Copy the output up to END OF REPORT.
"""
import os, sys, time, subprocess, urllib.request, traceback

ROOT = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset"
VARIANT = "A0"                      # "A0" ya "A1"
MV_SRC = "/kaggle/working/movinets_src"
MV_RAW = "https://raw.githubusercontent.com/Atze00/MoViNet-pytorch/main/movinets/"
T_FR, STRIDE, SIDE, CROP = 13, 6, 196, 172      # X3D jaisa clip (13 frames, stride 6), MoViNet-A0 ka native 172 resolution


def setup_movinet():
    for pkg in ("fvcore", "iopath", "einops", "onnx", "onnxruntime"):
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    d = os.path.join(MV_SRC, "movinets")
    os.makedirs(d, exist_ok=True)
    for f in ("__init__.py", "config.py", "models.py"):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(MV_RAW + f, p)
    if MV_SRC not in sys.path:
        sys.path.insert(0, MV_SRC)


def load_movinet():
    setup_movinet()
    from movinets import MoViNet
    from movinets.config import _C
    return MoViNet(getattr(_C.MODEL, "MoViNet" + VARIANT), causal=False, pretrained=True).eval()


import numpy as np, cv2, torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    """MoViNet -> 2048-d feature = conv7 + global pool + dense9 + Swish (X3D ke 2048-d feature jaisa position)."""
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, x):
        m = s.m
        x = m.avg(m.conv7(m.blocks(m.conv1(x))))
        x = m.classifier[1](m.classifier[0](x))
        return x.flatten(1)


def sample_clips(n_videos=12):
    vids = []
    for dp, _, fs in sorted(os.walk(ROOT)):
        for f in sorted(fs):
            if f.lower().endswith((".mp4", ".avi")):
                vids.append(os.path.join(dp, f))
    pick = [vids[int(i * len(vids) / n_videos)] for i in range(n_videos)]
    out = []
    span = (T_FR - 1) * STRIDE + 1
    for p in pick:
        cap = cv2.VideoCapture(p); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st = max(n // 2 - span // 2, 0); cap.set(cv2.CAP_PROP_POS_FRAMES, st); fr_list = []
        for off in range(span):
            ok, fr = cap.read()
            if not ok:
                break
            if off % STRIDE == 0:
                h, w = fr.shape[:2]; s = SIDE / min(h, w)
                fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s)))); h, w = fr.shape[:2]
                y0, x0 = (h - CROP) // 2, (w - CROP) // 2
                fr_list.append(cv2.cvtColor(fr[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_BGR2RGB))
        cap.release()
        if len(fr_list) == T_FR:
            out.append(np.stack(fr_list))
    return np.stack(out)                                   # N,T,H,W,3 uint8


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch", torch.__version__, "| device", dev)
    t0 = time.time(); m = load_movinet().to(dev)
    print(f"MoViNet-{VARIANT} (causal=False, Kinetics-600) load OK ({time.time() - t0:.0f} s) | params {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
    bb = Backbone(m).eval()

    x = torch.rand(2, 3, T_FR, CROP, CROP, device=dev)
    with torch.no_grad():
        f = bb(x); logits = m(x)
        logits2 = m.classifier[3](m.classifier[2](f[:, :, None, None, None])).flatten(1)
    print("feature shape", tuple(f.shape), "| wrapper vs pretrained classifier, max diff", float((logits - logits2).abs().max()))
    assert f.shape == (2, 2048) and float((logits - logits2).abs().max()) < 1e-3, "wrapper galat hai, ruk jao aur mujhe batao"

    clips = sample_clips()
    print("sample clips:", clips.shape)
    xs = torch.from_numpy(clips).to(dev).permute(0, 4, 1, 2, 3).float().div(255)
    print("\nNORMALIZATION check (K600 classifier ka confidence; sahi normalization par zyada confident hona chahiye)")
    for name, (mean, std) in {"[0,1]  (mean 0, std 1)": (0.0, 1.0), "(x-0.45)/0.225": (0.45, 0.225)}.items():
        with torch.no_grad():
            p = F.softmax(m((xs - mean) / std), dim=1)
        ent = -(p * (p + 1e-9).log()).sum(1).mean().item()
        print(f"  {name:24s} mean max-prob {p.max(1).values.mean().item():.3f} | mean entropy {ent:.2f}")

    # speed on this machine
    xb = torch.rand(1, 3, T_FR, CROP, CROP, device=dev)
    with torch.no_grad():
        for _ in range(3):
            bb(xb)
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            bb(xb)
        if dev == "cuda":
            torch.cuda.synchronize()
    print(f"\nGPU/torch: {(time.perf_counter() - t) / 20 * 1000:.1f} ms per clip (sirf andaaza)")

    # ONNX export + parity + CPU latency
    print("\nONNX export test")
    try:
        import onnxruntime as ort
        bbc = Backbone(load_movinet()).eval()
        xc = torch.rand(1, 3, T_FR, CROP, CROP)
        path = "/kaggle/working/mv_test.onnx"
        torch.onnx.export(bbc, xc, path, input_names=["clip"], output_names=["feature"], opset_version=17, dynamo=False)
        print(f"  export OK | {os.path.getsize(path) / 1e6:.1f} MB")
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = bbc(xc).numpy()
        out = sess.run(None, {"clip": xc.numpy()})[0]
        print("  parity (torch vs ORT) max abs diff:", float(np.abs(out - ref).max()), "| shape", out.shape)
        for th in (1, 4):
            so = ort.SessionOptions(); so.intra_op_num_threads = th
            s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            for _ in range(3):
                s.run(None, {"clip": xc.numpy()})
            ts = []
            for _ in range(15):
                t = time.perf_counter(); s.run(None, {"clip": xc.numpy()}); ts.append((time.perf_counter() - t) * 1000)
            print(f"  Kaggle CPU, {th} thread: median {np.median(ts):.0f} ms, p95 {np.percentile(ts, 95):.0f} ms")
    except Exception:
        print("  ONNX EXPORT/PARITY FAIL:")
        traceback.print_exc()
    print("\nEND OF REPORT")


main()
