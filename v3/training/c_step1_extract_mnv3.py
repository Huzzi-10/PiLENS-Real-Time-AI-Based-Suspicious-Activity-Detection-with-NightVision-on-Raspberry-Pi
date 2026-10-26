"""PiLENS CNN baseline Step 1: per-frame features from MobileNetV3-Large (ImageNet), a 2D CNN, for the CNN+LSTM / CNN-Transformer baselines.

Paste into one Kaggle GPU notebook cell (UCF-Crime dataset added, Internet ON).
Clips exactly as for X3D/MoViNet: 32 segments per video, 13 frames (stride 6) from the middle of each segment.
Each frame: resize to 232, 224x224 centre crop, ImageNet mean/std, MobileNetV3-Large -> 960-d.
Output: /kaggle/working/feats_mnv3l/<video>.npy  -> float16, shape (32, 13, 960)   (~0.8 MB/video, ~1.5 GB total)
        /kaggle/working/video_meta.csv
Resumes: features that already exist are skipped.
"""
import os, sys, time, csv, subprocess, collections

LIMIT = None         # None = poora chalao. Resume karta hai.
ROOT = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset"
T_FR, STRIDE_C, SIDE_C, CROP_C = 13, 6, 232, 224
MEAN_C, STD_C = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TAG = "mnv3l"
FDIM = 960

OUT = f"/kaggle/working/feats_{TAG}"
META = "/kaggle/working/video_meta.csv"
K, T, STRIDE, SIDE, CROP = 32, T_FR, STRIDE_C, SIDE_C, CROP_C
WORKERS, BATCH = 4, 16

import numpy as np, cv2, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

cv2.setNumThreads(0)


class Backbone(nn.Module):
    """MobileNetV3-Large (ImageNet) -> har frame ka 960-d feature (features + global avg pool, classifier se pehle)."""
    def __init__(s):
        super().__init__()
        import torchvision.models as tvm
        w = getattr(tvm.MobileNet_V3_Large_Weights, "IMAGENET1K_V2", tvm.MobileNet_V3_Large_Weights.DEFAULT)
        m = tvm.mobilenet_v3_large(weights=w)
        s.features, s.pool = m.features, m.avgpool
    def forward(s, x):                       # x: N,3,224,224 (normalized)
        return s.pool(s.features(x)).flatten(1)


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
                        last = cv2.cvtColor(fr[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_BGR2RGB)
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
    print("unique videos:", len(items))
    if LIMIT:
        items = items[:LIMIT]
    done = {f[:-4] for f in os.listdir(OUT) if f.endswith(".npy")}
    todo = [(n, p) for n, p in items if os.path.splitext(n)[0] not in done]
    print(f"features pehle se bane: {len(items) - len(todo)} | ab banane hain: {len(todo)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = Backbone().eval().to(dev)
    mean = torch.tensor(MEAN_C, device=dev).view(1, 3, 1, 1); std = torch.tensor(STD_C, device=dev).view(1, 3, 1, 1)
    print("SCRIPT VERSION: c_step1 v1 (MobileNetV3-Large per-frame 960-d) | LIMIT:", LIMIT)

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
            k_, t_ = x.shape[0], x.shape[1]
            x = x.reshape(k_ * t_, CROP, CROP, 3).permute(0, 3, 1, 2).float().div(255)
            x = ((x - mean) / std).contiguous()                         # K*T,3,224,224
            feats = []
            for b in range(0, x.shape[0], 128):
                with torch.autocast("cuda", enabled=(dev == "cuda")):
                    feats.append(m(x[b:b + 128]).float().cpu())
            f = torch.cat(feats).reshape(k_, t_, FDIM).numpy().astype(np.float16)   # K,T,960
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
        if a.shape != (K, T, FDIM) or not np.isfinite(a).all() or not a.any():
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
