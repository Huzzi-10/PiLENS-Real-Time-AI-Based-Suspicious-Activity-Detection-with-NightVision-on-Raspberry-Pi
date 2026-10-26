"""PiLENS MoViNet Step 1: 32 segment features per untrimmed video from MoViNet-A0 (Kinetics-600), frozen.
Paste into one Kaggle cell (after m_step0_check.py passes).
Every full video is split into 32 equal segments. From the middle of each segment one clip is taken
(13 frames, stride 6, about 2.4 s), resized to 196, centre-cropped to 172x172, pixels in [0,1] (no mean/std).
Output: /kaggle/working/feats_movinet_a0_v2/<video>.npy  -> float16, shape (32, 2048)
        /kaggle/working/video_meta.csv                   -> name, path, frames, fps of every video
Resumes: features that already exist are skipped.
Quick test first with LIMIT=8 (set below), then LIMIT=None.
Environment: PILENS_VARIANT=A0|A1, PILENS_COLOR=rgb|gray|irsim (simulated night IR).
"""
import os, sys, time, csv, subprocess, collections, urllib.request, zlib

LIMIT = None         # poora chalao (pipeline pehle verify ho chuka)
ROOT = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset"
VARIANT = os.environ.get("PILENS_VARIANT", "A0")                      # "A0" ya "A1" (m_step0 jaisa hi rakho)
MV_SRC = "/kaggle/working/movinets_src"
MV_RAW = "https://raw.githubusercontent.com/Atze00/MoViNet-pytorch/main/movinets/"
T_FR, STRIDE_MV, SIDE_MV, CROP_MV, MEAN_MV, STD_MV = 13, 6, 196, 172, 0.0, 1.0   # m_diag3: [0,1] sahi (train-CV probe 0.91 vs 0.53 for (x-0.45)/0.225)
TAG = "movinet_" + VARIANT.lower() + "_v2"      # v2 = pixels [0,1] (purani v1 (x-0.45)/0.225 galat thi)
TAG = os.environ.get("PILENS_TAG", TAG)          # night-vision run: movinet_a0_gray / movinet_a0_irsim
COLOR = os.environ.get("PILENS_COLOR", "rgb")     # rgb | gray | irsim (simulated night IR)


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

OUT = f"/kaggle/working/feats_{TAG}"
META = "/kaggle/working/video_meta.csv" if COLOR == "rgb" else f"/kaggle/working/video_meta_{TAG}.csv"
TEST_ONLY = os.environ.get("PILENS_TEST_ONLY") == "1"   # sirf 290 test videos (irsim)
K, T, STRIDE, SIDE, CROP = 32, 13, STRIDE_MV, SIDE_MV, CROP_MV
WORKERS, BATCH = 4, 16

setup_movinet()
import numpy as np, cv2, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

cv2.setNumThreads(0)


class Backbone(nn.Module):
    """MoViNet -> 2048-d feature = conv7 + global pool + dense9 + Swish."""
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, x):
        m = s.m
        x = m.avg(m.conv7(m.blocks(m.conv1(x))))
        x = m.classifier[1](m.classifier[0](x))
        return x.flatten(1)


def to_input(bgr, rng):
    """rgb: normal. gray: grayscale 3 channel mein (NoIR/IR raat ki video jaisi, rang nahi).
    irsim: simulated raat ki IR video: grayscale + kam contrast + blur + sensor noise (sirf test ke liye)."""
    if COLOR == "rgb":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if COLOR == "irsim":
        g = cv2.GaussianBlur(g, (3, 3), 0)
        g = 0.55 * g + 20 + rng.normal(0, 8, g.shape)
    g = np.clip(g, 0, 255).astype(np.uint8)
    return np.repeat(g[:, :, None], 3, 2)


def clip_starts(n, k=K, t=T, s=STRIDE):
    """Har segment ke beech ka clip: (start frame, kitne frames ka span)."""
    span = (t - 1) * s + 1
    out = []
    for i in range(k):
        c = int((i + 0.5) * n / k)
        out.append(min(max(c - span // 2, 0), max(n - span, 0)))
    return out, span


def list_videos():
    by_name = collections.OrderedDict()
    for dp, _, fs in sorted(os.walk(ROOT)):
        for f in sorted(fs):
            if f.lower().endswith((".mp4", ".avi", ".mkv")) and f not in by_name:
                by_name[f] = os.path.join(dp, f)       # duplicate naam par pehla path
    return list(by_name.items())


class VideoClips(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, path = self.items[i]
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if n <= 0 or not cap.isOpened():
            cap.release()
            return dict(name=name, path=path, n=0, fps=fps, bad=-1, x=None)
        starts, span = clip_starts(n)
        keep = set(range(0, span, STRIDE))
        clips, bad = [], 0
        last = np.zeros((CROP, CROP, 3), np.uint8)
        rng = np.random.RandomState(zlib.crc32(name.encode()))
        for st in starts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st)
            fr_list = []
            for off in range(span):
                if off in keep:
                    ok, fr = cap.read()
                    if ok:
                        h, w = fr.shape[:2]
                        s = SIDE / min(h, w)
                        fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))))
                        h, w = fr.shape[:2]
                        y0, x0 = (h - CROP) // 2, (w - CROP) // 2
                        last = to_input(fr[y0:y0 + CROP, x0:x0 + CROP], rng)
                    else:
                        bad += 1                        # padhne mein fail: pichla frame dohrao
                    fr_list.append(last)
                else:
                    cap.grab()
            clips.append(np.stack(fr_list))
        cap.release()
        return dict(name=name, path=path, n=n, fps=fps, bad=bad, x=torch.from_numpy(np.stack(clips)))


def main():
    os.makedirs(OUT, exist_ok=True)
    items = list_videos()
    if TEST_ONLY:
        tl = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Anomaly_Detection_splits", "Anomaly_Test.txt")
        keep = {os.path.splitext(os.path.basename(l.split()[0]))[0] for l in open(tl, errors="ignore") if l.strip()}
        items = [(n, p) for n, p in items if os.path.splitext(n)[0] in keep]
        print("TEST_ONLY:", len(items), "test videos")
    print("COLOR:", COLOR, "| TAG:", TAG)
    print("unique videos:", len(items))
    if LIMIT:
        items = items[:LIMIT]
    done = {f[:-4] for f in os.listdir(OUT) if f.endswith(".npy")}
    todo = [(n, p) for n, p in items if os.path.splitext(n)[0] not in done]
    print(f"features pehle se bane: {len(items) - len(todo)} | ab banane hain: {len(todo)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = Backbone(load_movinet()).eval().to(dev)

    new_meta = not os.path.exists(META) or os.path.getsize(META) == 0
    mf = open(META, "a", newline="")
    mw = csv.writer(mf)
    if new_meta:
        mw.writerow(["name", "path", "frames", "fps", "bad_reads"])

    dl = DataLoader(VideoClips(todo), batch_size=None, num_workers=WORKERS, prefetch_factor=2 if WORKERS else None)
    t0, nbad = time.time(), 0
    with torch.no_grad():
        for i, d in enumerate(dl, 1):
            if d["x"] is None:
                print("PADHNE MEIN FAIL:", d["name"])
                nbad += 1
                continue
            x = d["x"].to(dev, non_blocking=True)                       # K,T,H,W,3 uint8
            x = (x.float().div(255) - MEAN_MV) / STD_MV
            x = x.permute(0, 4, 1, 2, 3).contiguous()                   # K,3,T,H,W
            feats = []
            for b in range(0, x.shape[0], BATCH):
                feats.append(m(x[b:b + BATCH]).float().cpu())
            f = torch.cat(feats).numpy().astype(np.float16)             # K,2048
            np.save(os.path.join(OUT, os.path.splitext(d["name"])[0] + ".npy"), f)
            mw.writerow([d["name"], d["path"], d["n"], f"{d['fps']:.3f}", d["bad"]])
            mf.flush()
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)} | {el / i:.1f} s/video | baaki lagbhag {(len(todo) - i) * el / i / 60:.0f} min")
    mf.close()

    # Validation
    files = [f for f in os.listdir(OUT) if f.endswith(".npy")]
    bad = 0
    for f in files:
        a = np.load(os.path.join(OUT, f))
        if a.shape != (K, 2048) or not np.isfinite(a).all() or not a.any():
            bad += 1
            print("GALAT:", f, a.shape)
    print(f"\nfeature files valid: {len(files) - bad}/{len(files)}")
    if files:
        a = np.load(os.path.join(OUT, files[0]))
        print("sample shape:", a.shape, "dtype:", a.dtype, "mean: %.4f" % a.astype(np.float32).mean())
    if LIMIT:
        print("QUICK TEST OK" if bad == 0 and files else "QUICK TEST FAIL")
    print("END OF REPORT")


main()
