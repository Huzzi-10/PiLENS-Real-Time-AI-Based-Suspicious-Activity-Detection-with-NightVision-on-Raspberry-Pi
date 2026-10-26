"""PiLENS XD: second dataset (XD-Violence, Kaggle "XD-Violence Single-Label Edition", bypktt/xd-violence).
One Kaggle cell (GPU T4, Internet ON: annotations and CLIP are downloaded). Same protocol as on UCF-Crime:
  A. Features: 32 segments per video -> MoViNet-A0 (2048-d, as in m_step1) + CLIP ViT-B/16 (512-d, mean of 4 frames). One decode.
     Resumes: features are saved one video at a time. If the run hits the time limit, add this notebook's OUTPUT as input and rerun.
  B. Binary (official train/test, val = 15% of train, 5 seeds, test reported once): A0 mlp (Pi head), A0 tconv, CLIP mlp (large reference).
     Metrics: frame AP (XD's official metric), frame AUC, video AUC/AP, recall / normal FPR at the val-chosen threshold.
  C. Paired bootstrap (resampling videos): CLIP vs A0, tconv vs mlp.
  D. Cascade (A0 alert -> CLIP confirm around the A0 peak), thresholds from val only: fpr5 / fpr10 / youden.
  E. Crime type: 6 classes (Fighting, Shooting, Riot, Abuse, CarAccident, Explosion) on anomaly videos, A0 vs CLIP.
  F. Everything saved to /kaggle/working/download/xd_results.zip.
Note: this is the 753-video single-label subset of the official 800-video test set, so literature numbers are not strictly comparable.
Test annotations: official file https://roc-ng.github.io/XD-Violence/images/annotations.txt (frame start/end pairs).
"""
import os, sys, json, time, copy, csv, subprocess, collections, urllib.request, zipfile, shutil
import numpy as np

SCRIPT_VERSION = "xd v1.2 (2026-10-14)"
QUICK = os.environ.get("PILENS_QUICK") == "1"          # sirf sandbox test ke liye; Kaggle par isay mat chhedo
ROOT = os.environ.get("PILENS_ROOT", "/kaggle/input/datasets/bypktt/xd-violence")
WORK = os.environ.get("PILENS_WORK", "/kaggle/working")
ANN_URL = "https://roc-ng.github.io/XD-Violence/images/annotations.txt"
CLIP_NAME = "openai/clip-vit-base-patch16"
CLIP_RANDOM = os.environ.get("PILENS_CLIP_RANDOM") == "1"   # sandbox test (bina internet) - Kaggle par kabhi nahi
A0_TAG, CLIP_TAG = "xd_movinet_a0", "xd_clip_b16"
OUT = os.path.join(WORK, "xd_out")
MAX_HOURS = float(os.environ.get("PILENS_MAX_HOURS", "10.5"))   # Kaggle 12 h limit se pehle features save karke ruk jao
K = 32
T_FR, STRIDE_MV, SIDE_MV, CROP_MV = 13, 6, 196, 172          # m_step1 jaisa (A0 input [0,1])
CLIP_PICK, CLIP_SIDE = (0, 4, 8, 12), 224                     # 13 mein se 4 frames CLIP ke liye
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
WORKERS = int(os.environ.get("PILENS_WORKERS", "4"))

SEEDS = [0, 1] if QUICK else [0, 1, 2, 3, 4]
EPOCHS = 3 if QUICK else 40                 # binary (m_step2 jaisa)
CEPOCHS = 3 if QUICK else 80                # classes (m_step3 jaisa)
BAG, LR, WD, TOPK = (4 if QUICK else 30), 5e-4, 1e-4, 3
L_SMOOTH, L_SPARSE = 8e-5, 8e-5
VAL_FRAC, SPLIT_SEED = 0.15, 0
CBATCH, CLR, CWD = 32, 1e-3, 1e-2
NBOOT = 200 if QUICK else 2000
T_START = time.time()

print("SCRIPT VERSION:", SCRIPT_VERSION, "| MODE:", "QUICK (sirf test!)" if QUICK else "FULL")


def pip_need(*pkgs):
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


def _find(default, name):
    """Pehle /kaggle/working, warna /kaggle/input mein purane notebook ka output (dataset folder skip)."""
    if os.path.exists(default):
        return default
    if os.path.isdir("/kaggle/input"):
        for dp, dn, fn in os.walk("/kaggle/input"):
            if "ufc-crime-full-dataset" in dp or dp.startswith(ROOT) or "xd-violence" in dp:
                dn[:] = []
                continue
            if name in dn:
                return os.path.join(dp, name)
    return default


# =====================================================================================
# A. FEATURES (A0 + CLIP, ek decode)
# =====================================================================================
def setup_movinet():
    pip_need("fvcore", "iopath", "einops")
    src = os.path.join(WORK, "movinets_src")
    d = os.path.join(src, "movinets")
    os.makedirs(d, exist_ok=True)
    for f in ("__init__.py", "config.py", "models.py"):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve("https://raw.githubusercontent.com/Atze00/MoViNet-pytorch/main/movinets/" + f, p)
    if src not in sys.path:
        sys.path.insert(0, src)
    from movinets import MoViNet
    from movinets.config import _C
    return MoViNet(_C.MODEL.MoViNetA0, causal=False, pretrained=not CLIP_RANDOM).eval()   # random sirf sandbox test


def list_videos():
    by_name = collections.OrderedDict()
    for dp, _, fs in sorted(os.walk(ROOT)):
        for f in sorted(fs):
            if f.lower().endswith((".mp4", ".avi", ".mkv")) and f not in by_name:
                by_name[f] = os.path.join(dp, f)
    return list(by_name.items())


def clip_starts(n):
    span = (T_FR - 1) * STRIDE_MV + 1
    return [min(max(int((i + 0.5) * n / K) - span // 2, 0), max(n - span, 0)) for i in range(K)], span


def crop_sq(fr, side, crop):
    import cv2
    h, w = fr.shape[:2]
    s = side / min(h, w)
    fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))))
    h, w = fr.shape[:2]
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    return cv2.cvtColor(fr[y0:y0 + crop, x0:x0 + crop], cv2.COLOR_BGR2RGB)


def extract_features(need_names):
    import torch, cv2
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    cv2.setNumThreads(0)
    wa, wc = os.path.join(WORK, "feats_" + A0_TAG), os.path.join(WORK, "feats_" + CLIP_TAG)
    da = list(dict.fromkeys([wa, _find(wa, "feats_" + A0_TAG)]))      # working + (agar ho) purane notebook ka input
    dc = list(dict.fromkeys([wc, _find(wc, "feats_" + CLIP_TAG)]))
    for d in (wa, wc):
        os.makedirs(d, exist_ok=True)
    have = lambda ds, n: any(os.path.exists(os.path.join(d, n + ".npy")) for d in ds)
    items = [(f, p) for f, p in list_videos() if os.path.splitext(f)[0] in need_names]
    todo_a = {os.path.splitext(f)[0] for f, _ in items if not have(da, os.path.splitext(f)[0])}
    todo_c = {os.path.splitext(f)[0] for f, _ in items if not have(dc, os.path.splitext(f)[0])}
    print(f"A0 features {da} | pehle se {len(items) - len(todo_a)}, banane {len(todo_a)}")
    print(f"CLIP features {dc} | pehle se {len(items) - len(todo_c)}, banane {len(todo_c)}")
    todo = [(f, p) for f, p in items if os.path.splitext(f)[0] in (todo_a | todo_c)]
    if not todo:
        return da, dc, {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Backbone(nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, x):
            m = s.m
            x = m.avg(m.conv7(m.blocks(m.conv1(x))))
            return m.classifier[1](m.classifier[0](x)).flatten(1)

    a0 = Backbone(setup_movinet()).eval().to(dev) if todo_a else None
    cm = None
    if todo_c:
        pip_need("transformers")
        from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig
        if CLIP_RANDOM:
            cm = CLIPVisionModelWithProjection(CLIPVisionConfig(patch_size=16, projection_dim=512, num_hidden_layers=2))
            print("WARNING: CLIP random weights (sirf sandbox test)")
        else:
            cm = CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME)
        cm = cm.eval().to(dev)
        if dev == "cuda":
            cm = cm.half()
    cmean, cstd = torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)

    class VC(Dataset):
        def __init__(s, it):
            s.it = it
        def __len__(s):
            return len(s.it)
        def __getitem__(s, i):
            name, path = s.it[i]
            cap = cv2.VideoCapture(path)
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
            if n <= 0 or not cap.isOpened():
                cap.release(); return dict(name=name, n=0, fps=fps, bad=-1, a=None, c=None)
            starts, span = clip_starts(n)
            keep = set(range(0, span, STRIDE_MV))
            A, C, bad = [], [], 0
            la, lc = np.zeros((CROP_MV, CROP_MV, 3), np.uint8), np.zeros((CLIP_SIDE, CLIP_SIDE, 3), np.uint8)
            for st in starts:
                cap.set(cv2.CAP_PROP_POS_FRAMES, st)
                fa_, fc_, j = [], [], 0
                for off in range(span):
                    if off in keep:
                        ok, fr = cap.read()
                        if ok:
                            la = crop_sq(fr, SIDE_MV, CROP_MV)
                            if j in CLIP_PICK:
                                lc = crop_sq(fr, CLIP_SIDE, CLIP_SIDE)
                        else:
                            bad += 1
                        fa_.append(la)
                        if j in CLIP_PICK:
                            fc_.append(lc)
                        j += 1
                    else:
                        cap.grab()
                A.append(np.stack(fa_)); C.append(np.stack(fc_))
            cap.release()
            return dict(name=name, n=n, fps=fps, bad=bad, a=torch.from_numpy(np.stack(A)), c=torch.from_numpy(np.stack(C)))

    meta_p = os.path.join(WORK, "video_meta_xd.csv")
    new = not os.path.exists(meta_p)
    mf = open(meta_p, "a", newline=""); mw = csv.writer(mf)
    if new:
        mw.writerow(["name", "path", "frames", "fps", "bad_reads"])
    dl = DataLoader(VC(todo), batch_size=None, num_workers=WORKERS, prefetch_factor=2 if WORKERS else None)
    t0, frames = time.time(), {}
    with torch.no_grad():
        for i, d in enumerate(dl, 1):
            nm = os.path.splitext(d["name"])[0]
            if (time.time() - T_START) / 3600 > MAX_HOURS:
                print(f"TIME LIMIT ({MAX_HOURS} h): features save ho gaye, baqi agle run mein (OUTPUT ko Add Input karke dobara chalao)."); break
            if d["a"] is None:
                print("PADHNE MEIN FAIL:", d["name"])
                with open(os.path.join(WORK, "bad_videos_xd.txt"), "a") as bf:
                    bf.write(os.path.splitext(d["name"])[0] + "\n")
                continue
            frames[nm] = d["n"]
            if nm in todo_a:
                x = d["a"].to(dev).float().div(255).permute(0, 4, 1, 2, 3).contiguous()
                f = torch.cat([a0(x[b:b + 16]).float().cpu() for b in range(0, K, 16)]).numpy().astype(np.float16)
                assert f.shape == (K, 2048) and np.isfinite(f).all()
                np.save(os.path.join(wa, nm + ".npy"), f)
            if nm in todo_c:
                x = d["c"].to(dev).reshape(-1, CLIP_SIDE, CLIP_SIDE, 3).permute(0, 3, 1, 2).float().div(255)
                x = ((x - cmean) / cstd)
                x = x.half() if dev == "cuda" else x
                e = torch.cat([cm(pixel_values=x[b:b + 64]).image_embeds.float() for b in range(0, x.shape[0], 64)])
                e = e.reshape(K, len(CLIP_PICK), -1).mean(1).cpu().numpy().astype(np.float16)
                assert e.shape == (K, 512) and np.isfinite(e).all()
                np.save(os.path.join(wc, nm + ".npy"), e)
            mw.writerow([d["name"], "", d["n"], f"{d['fps']:.3f}", d["bad"]]); mf.flush()
            if i == 2 or i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"features {i}/{len(todo)} | {el / i:.1f} s/video | baaki lagbhag {(len(todo) - i) * el / i / 60:.0f} min", flush=True)
    mf.close()
    return da, dc, frames


# =====================================================================================
# metrics
# =====================================================================================
def auc_score(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); ranks = ((cum - cnt + 1) + cum) / 2.0
    r = ranks[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def wauc(s, pos, neg):
    """Weighted AUC: har unit (video segment) ke andar sab frames ka score same -> bilkul frame AUC ke barabar, lekin tez."""
    o = np.argsort(s, kind="mergesort"); s, pos, neg = s[o], pos[o], neg[o]
    _, idx, inv = np.unique(s, return_index=True, return_inverse=True)
    gp, gn = np.bincount(inv, weights=pos), np.bincount(inv, weights=neg)
    below = np.cumsum(gn) - gn
    P, N = gp.sum(), gn.sum()
    return float(((gp * (below + 0.5 * gn)).sum()) / (P * N)) if P > 0 and N > 0 else float("nan")


def ap_score(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    o = np.argsort(-s, kind="mergesort"); s, y = s[o], y[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    ends = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp, fp = tp[ends], fp[ends]
    prec, rec = tp / (tp + fp), tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, (2 * p * r / max(p + r, 1e-9))


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def confusion(y, p, n):
    cm = np.zeros((n, n), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


# =====================================================================================
# data lists
# =====================================================================================
def read_list(p):
    out = []
    for l in open(p, errors="ignore"):
        t = l.strip().split()
        if t:
            out.append((os.path.splitext(os.path.basename(t[0]))[0], t[0].split("/")[0]))
    return out


def read_meta(paths):
    out = {}
    for p in paths:
        if p and os.path.exists(p):
            for r in csv.reader(open(p)):
                if len(r) >= 3 and r[0] != "name":
                    try:
                        out[os.path.splitext(r[0])[0]] = int(r[2])
                    except ValueError:
                        pass
    return out


def read_ann(p):
    ann = {}
    for l in open(p, errors="ignore"):
        t = l.split()
        if len(t) >= 6:
            nm = os.path.splitext(t[0])[0]
            nums = [int(x) for x in t[2:6]]
            ann[nm] = [(nums[0], nums[1])] + ([(nums[2], nums[3])] if nums[2] >= 0 else []) if nums[0] >= 0 else []
    return ann


def units(n, segs):
    """Har segment mein kitne anomaly / normal frames (frame_scores mapping m_step2 jaisi)."""
    seg = np.minimum((np.arange(n) * K) // n, K - 1)
    g = np.zeros(n, np.uint8)
    for s, e in segs:
        g[max(s, 0):min(e + 1, n)] = 1
    pos = np.bincount(seg, weights=g, minlength=K)
    return pos, np.bincount(seg, minlength=K) - pos


# =====================================================================================
# B. binary heads
# =====================================================================================
def build(head, d):
    import torch.nn as nn
    if head == "mlp":
        return nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.6), nn.Linear(512, 32), nn.ReLU(), nn.Dropout(0.6), nn.Linear(32, 1))

    class TConv(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.Conv1d(d, 256, 3, padding=1), nn.ReLU(), nn.Dropout(0.6),
                                  nn.Conv1d(256, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.6), nn.Conv1d(64, 1, 1))
        def forward(s, x):
            return s.net(x.transpose(1, 2)).transpose(1, 2)
    return TConv()


def hidden_and_out(m, head, x):
    """Student ka hidden (KD feature ke liye) aur logit. Architecture purana hi (Pi export same)."""
    if head == "mlp":
        h = m[:2](x); return h, m[2:](h).squeeze(-1)
    h = m.net[:2](x.transpose(1, 2))
    return h.transpose(1, 2), m.net[2:](h).transpose(1, 2).squeeze(-1)


def seg_scores(m, X, dev):
    import torch
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(m(X[i:i + 128].to(dev)).squeeze(-1).double().cpu())
    return 1.0 / (1.0 + np.exp(-torch.cat(out).numpy()))


def train_bin(head, seed, Xtr, ytr, Xva, yva, dev, tgt=None, ctr=None, lam_s=0.0, lam_f=0.0):
    """m_step2 wala MIL training + optional KD. tgt: (N,K) teacher segment targets, ctr: (N,K,512) CLIP features."""
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    m = build(head, Xtr.shape[-1]).to(dev)
    params = list(m.parameters())
    proj = None
    if lam_f > 0:
        proj = nn.Linear(512 if head == "mlp" else 256, ctr.shape[-1]).to(dev); params += list(proj.parameters())
        Cg = F.normalize(ctr.to(dev), dim=-1)
    if lam_s > 0:
        Tg = tgt.to(dev).float()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    ia, ino = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    Xg = Xtr.to(dev)
    best, best_state = -1, None
    for ep in range(EPOCHS):
        m.train()
        pa, pn = rng.permutation(ia), rng.permutation(ino)
        for b in range(min(len(pa), len(pn)) // BAG):
            ba, bn = pa[b * BAG:(b + 1) * BAG], pn[b * BAG:(b + 1) * BAG]
            idx = np.r_[ba, bn]
            h, z = hidden_and_out(m, head, Xg[idx])
            s = torch.sigmoid(z); sa, sn = s[:BAG], s[BAG:]
            ta, tn = sa.topk(TOPK, 1).values.mean(1), sn.topk(TOPK, 1).values.mean(1)
            loss = F.relu(1 - ta[:, None] + tn[None, :]).mean() + L_SMOOTH * ((sa[:, 1:] - sa[:, :-1]) ** 2).sum(1).mean() + L_SPARSE * sa.sum(1).mean()
            if lam_s > 0:
                loss = loss + lam_s * F.binary_cross_entropy_with_logits(z, Tg[idx])
            if lam_f > 0:
                loss = loss + lam_f * (1 - (F.normalize(proj(h), dim=-1) * Cg[idx]).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = auc_score(yva, seg_scores(m, Xva, dev).max(1))
        if best_state is None or v > best:
            best, best_state = v, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return m, best


# =====================================================================================
# C. class heads (m_step3 jaisa mil_topk / linear)
# =====================================================================================
def build_cls(head, nc, d):
    import torch.nn as nn

    class H(nn.Module):
        def __init__(s):
            super().__init__(); s.drop, s.fc = nn.Dropout(0.5), nn.Linear(d, nc)
        def forward(s, x):
            x = s.drop(x)
            if head == "linear":
                return s.fc(x.mean(1))
            return s.fc(x).topk(min(TOPK, x.shape[1]), dim=1).values.mean(1)
    return H()


def train_cls(head, seed, Xtr, ytr, Xva, yva, Xte, nc, dev, wrs=False, soft=None, lam=0.0):
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    m = build_cls(head, nc, Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=CLR, weight_decay=CWD)
    Xg, yg = Xtr.to(dev), torch.from_numpy(ytr).to(dev)
    Sg = soft.to(dev).float() if lam > 0 else None
    w = 1.0 / np.bincount(ytr, minlength=nc)[ytr]; w /= w.sum()
    best, best_state = -1, None
    for ep in range(CEPOCHS):
        m.train()
        order = rng.choice(len(ytr), len(ytr), p=w) if wrs else rng.permutation(len(ytr))
        for b in range(0, len(order), CBATCH):
            i = order[b:b + CBATCH]
            sub = np.sort(rng.choice(K, K // 2, replace=False))
            lg = m(Xg[i][:, sub])
            loss = F.cross_entropy(lg, yg[i], label_smoothing=0.1)
            if lam > 0:
                loss = loss + lam * F.kl_div(F.log_softmax(lg, 1), Sg[i], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            f1 = macro_f1(yva, m(Xva.to(dev)).argmax(1).cpu().numpy(), nc)
        if f1 > best:
            best, best_state = f1, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        return m(Xte.to(dev)).cpu(), best, m


def strat_folds(y, nf, seed):
    rs = np.random.RandomState(seed); f = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rs.shuffle(idx)
        f[idx] = np.arange(len(idx)) % nf
    return f



def fill_frames(frames, names):
    miss = {n for n in names if n not in frames}
    if miss:
        import cv2
        for dp, _, fs in os.walk(ROOT):
            for f in fs:
                n = os.path.splitext(f)[0]
                if n in miss and f.lower().endswith((".mp4", ".avi", ".mkv")):
                    cap = cv2.VideoCapture(os.path.join(dp, f)); frames[n] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release(); miss.discard(n)
    assert not miss, f"frame count nahi mila: {sorted(miss)[:5]}"
    return frames


def op_thr(vmax, yv, kind):
    """Val par operating point: youden | fpr5 | fpr10 (fpr = normal videos ka val FPR limit, uske andar sab se zyada recall)."""
    cand = np.unique(vmax)
    npos, nneg = max((yv == 1).sum(), 1), max((yv == 0).sum(), 1)
    rf = lambda t: (((vmax >= t) & (yv == 1)).sum() / npos, ((vmax >= t) & (yv == 0)).sum() / nneg)
    if kind == "youden":
        return float(max(cand, key=lambda t: rf(t)[0] - rf(t)[1]))
    ok = [t for t in cand if rf(t)[1] <= {"fpr5": 0.05, "fpr10": 0.10}[kind]]
    return float(min(ok)) if ok else float(cand.max()) + 1e-9


def win_max(c, p, r):
    return np.array([c[i, max(int(q) - r, 0):int(q) + r + 1].max() for i, q in enumerate(p)])


def cascade_eval(aV, cV, yv, aT, cT, yt, kind, r):
    """A0 alert (val-chosen thr) -> CLIP confirm: A0 ke peak segment ke aas-paas (+-r segment) CLIP ka max score >= thrC (val par cascade Youden se)."""
    thrA = op_thr(aV.max(1), yv, kind)
    alertV, alertT = aV.max(1) >= thrA, aT.max(1) >= thrA
    wV, wT = win_max(cV, aV.argmax(1), r), win_max(cT, aT.argmax(1), r)
    npos, nneg = max((yv == 1).sum(), 1), max((yv == 0).sum(), 1)
    cand = np.unique(wV[alertV]) if alertV.any() else np.array([0.5])
    obj = lambda t: ((alertV & (wV >= t) & (yv == 1)).sum() / npos) - ((alertV & (wV >= t) & (yv == 0)).sum() / nneg)
    thrC = float(max(cand, key=obj))
    dec = alertT & (wT >= thrC)
    tp, tn = max((yt == 1).sum(), 1), max((yt == 0).sum(), 1)
    out = dict(thrA=thrA, thrC=thrC, rec_a0=float((alertT & (yt == 1)).sum() / tp), fpr_a0=float((alertT & (yt == 0)).sum() / tn),
               rec=float((dec & (yt == 1)).sum() / tp), fpr=float((dec & (yt == 0)).sum() / tn))
    return out, alertT



# =====================================================================================
# XD main
# =====================================================================================
CLASSES6 = ["Fighting", "Shooting", "Riot", "Abuse", "CarAccident", "Explosion"]


def find_root():
    """Dataset folder dhoondo: jis ke andar train/ aur test/ hon (aur un mein Normal/)."""
    global ROOT
    if os.path.isdir(os.path.join(ROOT, "train")):
        return ROOT
    for dp, dn, _ in os.walk("/kaggle/input"):
        if "train" in dn and "test" in dn and os.path.isdir(os.path.join(dp, "train", "Normal")):
            ROOT = dp
            return dp
    raise SystemExit(f"ROKO: XD-Violence dataset nahi mila ({ROOT}). Notebook mein 'Add Input' se bypktt/xd-violence lagao.")


def list_split(split):
    out = []
    base = os.path.join(ROOT, split)
    for c in sorted(os.listdir(base)):
        d = os.path.join(base, c)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".mp4", ".avi", ".mkv")):
                out.append((os.path.splitext(f)[0], 0 if c == "Normal" else 1, c))
    return out


def norm_name(s):
    """Kaggle naam 'Riot_v=abc___1_label_B4-0-0' <-> official 'v=abc__#1_label_B4-0-0'."""
    return s.replace("__#", "___").strip()


def load_annotations():
    p = os.path.join(WORK, "xd_annotations.txt")
    if not os.path.exists(p):
        for dp, _, fs in os.walk("/kaggle/input"):
            if dp.startswith(ROOT):
                continue
            for f in fs:
                if f.lower() == "annotations.txt":
                    shutil.copy(os.path.join(dp, f), p); break
            if os.path.exists(p):
                break
    if not os.path.exists(p):
        try:
            urllib.request.urlretrieve(ANN_URL, p)
        except Exception as e:
            print(f"WARNING: annotations download nahi hui ({e}). Internet ON karo. Frame-level metrics nahi banenge, sirf video-level.")
            return {}
    ann = {}
    for l in open(p, errors="ignore"):
        t = l.split()
        if len(t) >= 3:
            nums = [int(float(x)) for x in t[1:]]
            ann[norm_name(t[0][:-4] if t[0].lower().endswith(".mp4") else t[0])] = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    print(f"annotations: {len(ann)} anomaly test videos ({p})")
    return ann


def wap(s, pos, neg):
    """Weighted AP: har unit (segment) ke saare frames ka score same -> bilkul frame-level AP ke barabar (ties group)."""
    o = np.argsort(-s, kind="mergesort"); s, pos, neg = s[o], pos[o], neg[o]
    ends = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp, fp = np.cumsum(pos)[ends], np.cumsum(neg)[ends]
    P = pos.sum()
    if P <= 0:
        return float("nan")
    prec, rec = tp / np.maximum(tp + fp, 1e-12), tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    find_root()
    train_all, test = list_split("train"), list_split("test")
    assert not ({n for n, *_ in train_all} & {n for n, *_ in test}), "train/test overlap!"
    print(f"device {dev} | root {ROOT}")
    print(f"train {len(train_all)} (anomaly {sum(l for _, l, _ in train_all)}) | test {len(test)} (anomaly {sum(l for _, l, _ in test)})")
    print("train per class:", dict(collections.Counter(c for *_, c in train_all)))
    print("test per class:", dict(collections.Counter(c for *_, c in test)))
    assert 0 < sum(l for _, l, _ in train_all) < len(train_all), "labels galat (sab ek hi class)"

    FA, FC, fr_new = extract_features({n for n, *_ in train_all + test})
    has = lambda ds, n: any(os.path.exists(os.path.join(d, n + ".npy")) for d in ds)
    bad = set()
    for bp in dict.fromkeys([os.path.join(WORK, "bad_videos_xd.txt"), _find(os.path.join(WORK, "bad_videos_xd.txt"), "bad_videos_xd.txt")]):
        if os.path.exists(bp):
            bad |= {l.strip() for l in open(bp) if l.strip()}
    miss = [n for n, *_ in train_all + test if not (has(FA, n) and has(FC, n))]
    bad_miss = [n for n in miss if n in bad]
    if bad_miss:
        assert len(bad_miss) <= max(3, 0.01 * (len(train_all) + len(test))), "bohat saari videos corrupt"
        print(f"CORRUPT videos (file kharab, padhi nahi ja sakti) chhor di: {bad_miss}")
        bt = [n for n in bad_miss if n in {t[0] for t in test}]
        if bt:
            print(f"  NOTE: in mein se {len(bt)} test ki hain -> test {len(test)} se {len(test) - len(bt)} videos (paper mein likhna)")
        train_all = [t for t in train_all if t[0] not in bad_miss]
        test = [t for t in test if t[0] not in bad_miss]
        miss = [n for n in miss if n not in bad_miss]
    if miss:
        print(f"\nFEATURES ABHI ADHOORE: {len(miss)} videos baqi. Ye ghalti nahi hai (time limit).")
        print("Karo: is version ka OUTPUT notebook mein 'Add Input' karo, phir wahi cell dobara 'Save & Run All'. Bane hue features skip honge.")
        print("END OF REPORT (partial)")
        return
    frames = read_meta([os.path.join(WORK, "video_meta_xd.csv"), _find(os.path.join(WORK, "video_meta_xd.csv"), "video_meta_xd.csv")])
    frames.update(fr_new)
    frames = fill_frames(frames, [n for n, *_ in test])

    ann_raw = load_annotations()
    ann, unmatched = {}, []
    for n, l, c in test:
        if l == 0:
            ann[n] = []
            continue
        key = norm_name(n.split("_", 1)[1]) if n.startswith(c + "_") else norm_name(n)
        if key in ann_raw:
            ann[n] = ann_raw[key]
        elif norm_name(n) in ann_raw:
            ann[n] = ann_raw[norm_name(n)]
        else:
            unmatched.append(n)
    n_anom = sum(l for _, l, _ in test)
    FRAME_OK = len(ann_raw) > 0 and len(unmatched) <= 0.1 * n_anom
    print(f"annotation match: {n_anom - len(unmatched)}/{n_anom} anomaly test videos" + (f" | unmatched e.g. {unmatched[:3]}" if unmatched else ""))
    if not FRAME_OK:
        print("WARNING: frame-level metrics band (annotations match nahi hui). Sirf video-level report hoga.")
    fte = [i for i, (n, *_) in enumerate(test) if n in ann]          # frame metrics in videos par (unmatched anomaly bahar)

    rs = np.random.RandomState(SPLIT_SEED); tr, va = [], []
    for lab in (0, 1):
        idx = [i for i, (_, l, _) in enumerate(train_all) if l == lab]; rs.shuffle(idx)
        nv = int(round(len(idx) * VAL_FRAC)); va += idx[:nv]; tr += idx[nv:]
    def load(ds, names):
        pick = lambda n: next(os.path.join(d, n + ".npy") for d in ds if os.path.exists(os.path.join(d, n + ".npy")))
        return torch.from_numpy(np.stack([np.load(pick(n)) for n in names]).astype(np.float32))
    ntr, nva, nte = [train_all[i][0] for i in tr], [train_all[i][0] for i in va], [n for n, *_ in test]
    ytr, yva = np.array([train_all[i][1] for i in tr]), np.array([train_all[i][1] for i in va])
    yte = np.array([l for _, l, _ in test])

    def normed(d):
        Xtr, Xva, Xte = load(d, ntr), load(d, nva), load(d, nte)
        D = Xtr.shape[-1]
        mu, sd = Xtr.reshape(-1, D).mean(0), Xtr.reshape(-1, D).std(0) + 1e-6
        return [(x - mu) / sd for x in (Xtr, Xva, Xte)]
    Atr, Ava, Ate = normed(FA)
    Ctr, Cva, Cte = normed(FC)
    print(f"train {len(tr)} | val {len(va)} | test {len(nte)} | A0 {tuple(Atr.shape)} | CLIP {tuple(Ctr.shape)}")

    U = [units(frames[n], ann.get(n, [])) for n in nte]
    UP, UN = np.stack([u[0] for u in U]), np.stack([u[1] for u in U])
    fidx = np.array(fte)

    pack = dict(test_y=yte, val_y=yva, units_pos=UP, units_neg=UN, test_names=np.array(nte), frame_idx=fidx)
    R = {"info": {}, "binary": {}, "bootstrap": {}, "cascade": {}, "classes": {}}
    R["info"] = dict(script=SCRIPT_VERSION, root=ROOT, n_train=len(train_all), n_val=len(va), n_test=len(test), n_test_anom=int(n_anom),
                     ann_matched=int(n_anom - len(unmatched)), frame_metrics=bool(FRAME_OK), seeds=SEEDS)

    def fmet(ts, ix=None):
        ix = fidx if ix is None else ix
        if not FRAME_OK:
            return float("nan"), float("nan")
        s, p, q = ts[ix].ravel(), UP[ix].ravel(), UN[ix].ravel()
        return wap(s, p, q), wauc(s, p, q)

    def evaluate(m, Xva_, Xte_, bval):
        vs, ts = seg_scores(m, Xva_, dev), seg_scores(m, Xte_, dev)
        thr = op_thr(vs.max(1), yva, "youden")
        pred = (ts.max(1) >= thr).astype(int)
        p, r, f1 = prf(yte, pred)
        fap, fauc = fmet(ts)
        return dict(val_auc=bval, frame_ap=fap, frame_auc=fauc, video_auc=auc_score(yte, ts.max(1)), video_ap=ap_score(yte, ts.max(1)),
                    thr=float(thr), prec=p, rec=r, f1=f1, normal_fpr=float(((pred == 1) & (yte == 0)).sum() / max((yte == 0).sum(), 1))), vs, ts

    def run_rows(key, head, Xtr_, Xva_, Xte_):
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            m, bv = train_bin(head, seed, Xtr_, ytr, Xva_, yva, dev)
            row, vs, ts = evaluate(m, Xva_, Xte_, bv)
            rows.append(dict(seed=seed, **row)); pack[f"val_{key}_s{seed}"] = vs; pack[f"test_{key}_s{seed}"] = ts
            if seed == 0 and key == "a0_mlp":
                torch.save(m.state_dict(), os.path.join(OUT, f"{key}_seed0.pt"))
            print(f"  [{key} s{seed}] {time.time() - t0:.0f}s | val {bv:.3f} | frame AP {row['frame_ap']:.4f} | frame AUC {row['frame_auc']:.4f} | video AUC {row['video_auc']:.3f} | rec {row['rec']:.3f} FPR {row['normal_fpr']:.3f}", flush=True)
        R["binary"][key] = rows
        ms = lambda k: f"{np.mean([r[k] for r in rows]):.4f} +- {np.std([r[k] for r in rows]):.4f}"
        print(f"  == {key}: frame AP {ms('frame_ap')} | frame AUC {ms('frame_auc')} | video AUC {ms('video_auc')} | video AP {ms('video_ap')}")

    print("\n=== B. BINARY (5 seeds; threshold val Youden se) ===")
    for key, head, X in (("a0_mlp", "mlp", (Atr, Ava, Ate)), ("a0_tconv", "tconv", (Atr, Ava, Ate)), ("clip_mlp", "mlp", (Ctr, Cva, Cte))):
        run_rows(key, head, *X)

    print(f"\n=== C. paired bootstrap ({NBOOT}x, videos resample, seed-mean scores) ===")
    rng = np.random.RandomState(0)
    BS = [rng.randint(0, len(yte), len(yte)) for _ in range(NBOOT)]
    avg = lambda key: np.mean([pack[f"test_{key}_s{s}"] for s in SEEDS], 0)
    fset = set(fidx.tolist())
    def boot(k1, k2):
        a, b = avg(k1), avg(k2); dv, df = [], []
        for ix in BS:
            dv.append(auc_score(yte[ix], a[ix].max(1)) - auc_score(yte[ix], b[ix].max(1)))
            if FRAME_OK:
                jx = np.array([i for i in ix if i in fset])
                df.append(fmet(a, jx)[0] - fmet(b, jx)[0])
        o = dict(video_auc_diff=auc_score(yte, a.max(1)) - auc_score(yte, b.max(1)), video_ci=[float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))])
        if FRAME_OK:
            o.update(frame_ap_diff=fmet(a)[0] - fmet(b)[0], frame_ap_ci=[float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))])
        R["bootstrap"][f"{k1}__vs__{k2}"] = o
        print(f"  {k1} vs {k2}: video AUC {o['video_auc_diff']:+.4f} [{o['video_ci'][0]:+.4f}, {o['video_ci'][1]:+.4f}]" +
              (f" | frame AP {o['frame_ap_diff']:+.4f} [{o['frame_ap_ci'][0]:+.4f}, {o['frame_ap_ci'][1]:+.4f}]" if FRAME_OK else ""))
    boot("clip_mlp", "a0_mlp"); boot("a0_tconv", "a0_mlp")

    print("\n=== D. CASCADE: A0 alert -> CLIP confirm (mlp heads; thresholds sirf val se; primary fpr5 r=0) ===")
    for k in ("fpr5", "fpr10", "youden"):
        for r in (0, 1):
            rows = []
            for s in SEEDS:
                o, _ = cascade_eval(pack[f"val_a0_mlp_s{s}"], pack[f"val_clip_mlp_s{s}"], yva, pack[f"test_a0_mlp_s{s}"], pack[f"test_clip_mlp_s{s}"], yte, k, r)
                rows.append(dict(seed=s, **o))
            R["cascade"][f"{k}_r{r}"] = rows
            f = lambda x: f"{100 * np.mean([q[x] for q in rows]):.1f} +- {100 * np.std([q[x] for q in rows]):.1f}"
            print(f"  gate {k:7s} r={r} | A0 alone: recall {f('rec_a0')} FPR {f('fpr_a0')} | A0+CLIP: recall {f('rec')} FPR {f('fpr')}")

    print("\n=== E. CRIME TYPE: 6 classes (anomaly videos; train ke 15% val; macro F1 se model chuna) ===")
    dtr_all = [(n, c) for n, l, c in train_all if l == 1]; dte = [(n, c) for n, l, c in test if l == 1]
    dcls = [c for c in CLASSES6 if any(k == c for _, k in dtr_all)] + sorted({c for _, c in dtr_all} - set(CLASSES6))
    cid = {c: i for i, c in enumerate(dcls)}; nc = len(dcls)
    dte = [(n, c) for n, c in dte if c in cid]
    rs = np.random.RandomState(7); dtr, dva = [], []
    for c in dcls:
        idx = [i for i, (_, k) in enumerate(dtr_all) if k == c]; rs.shuffle(idx)
        nv = max(1, int(round(len(idx) * VAL_FRAC))); dva += idx[:nv]; dtr += idx[nv:]
    def cnormed(d):
        Xtr, Xva, Xte = load(d, [dtr_all[i][0] for i in dtr]), load(d, [dtr_all[i][0] for i in dva]), load(d, [n for n, _ in dte])
        D = Xtr.shape[-1]; mu, sd = Xtr.reshape(-1, D).mean(0), Xtr.reshape(-1, D).std(0) + 1e-6
        return [(x - mu) / sd for x in (Xtr, Xva, Xte)]
    ytr_c = np.array([cid[dtr_all[i][1]] for i in dtr]); yva_c = np.array([cid[dtr_all[i][1]] for i in dva]); yte_c = np.array([cid[c] for _, c in dte])
    print(f"classes {dcls} | train {len(dtr)} val {len(dva)} test {len(dte)} | test per class {dict(collections.Counter(c for _, c in dte))}")
    maj = float(np.mean(yte_c == np.bincount(ytr_c, minlength=nc).argmax()))
    R["classes"]["info"] = dict(classes=dcls, chance=1.0 / nc, majority=maj, n_test=len(dte))
    for who, d in (("a0", FA), ("clip", FC)):
        X = cnormed(d)
        for head in ("mil_topk", "linear"):
            rows, cms = [], []
            for s in SEEDS:
                lg, bv, _ = train_cls(head, s, X[0], ytr_c, X[1], yva_c, X[2], nc, dev)
                pr = lg.argmax(1).numpy(); top2 = np.argsort(-lg.numpy(), 1)[:, :2]
                rows.append(dict(seed=s, val_f1=bv, acc=float((pr == yte_c).mean()), macro_f1=macro_f1(yte_c, pr, nc),
                                 top2=float(np.mean([yte_c[i] in top2[i] for i in range(len(yte_c))]))))
                cms.append(confusion(yte_c, pr, nc)); pack[f"cls_{who}_{head}_s{s}"] = lg.numpy()
            R["classes"][f"{who}_{head}"] = rows
            np.savetxt(os.path.join(OUT, f"confusion_{who}_{head}_seed0.csv"), cms[0], fmt="%d", delimiter=",", header=",".join(dcls))
            f = lambda k: f"{100 * np.mean([q[k] for q in rows]):.1f} +- {100 * np.std([q[k] for q in rows]):.1f}"
            print(f"  {who:4s} {head:8s}: acc {f('acc')} | macro F1 {f('macro_f1')} | top-2 {f('top2')}  (chance {100 / nc:.1f}, majority {100 * maj:.1f})")
    pack["cls_test_y"] = yte_c

    # ---- F. save ----
    np.savez_compressed(os.path.join(OUT, "xd_pack.npz"), **pack)
    json.dump(R, open(os.path.join(OUT, "xd_results.json"), "w"), indent=1, default=float)
    L = [f"# XD-Violence results ({SCRIPT_VERSION})", "",
         f"Dataset: Kaggle XD-Violence Single-Label Edition | train {len(train_all)} | test {len(test)} (anomaly {n_anom}) | annotation match {n_anom - len(unmatched)}/{n_anom}", "",
         "| model | frame AP | frame AUC | video AUC | video AP | recall | normal FPR |", "|---|---|---|---|---|---|---|"]
    for k, rows in R["binary"].items():
        g = lambda x: f"{np.mean([r[x] for r in rows]):.4f} +- {np.std([r[x] for r in rows]):.4f}"
        L.append(f"| {k} | {g('frame_ap')} | {g('frame_auc')} | {g('video_auc')} | {g('video_ap')} | {g('rec')} | {g('normal_fpr')} |")
    L += ["", "Bootstrap:"] + [f"- {k}: {json.dumps(v)}" for k, v in R["bootstrap"].items()]
    L += ["", "Cascade (mean over seeds, %):"]
    for k, rows in R["cascade"].items():
        g = lambda x: f"{100 * np.mean([q[x] for q in rows]):.1f}"
        L.append(f"- {k}: A0 recall {g('rec_a0')} FPR {g('fpr_a0')} -> A0+CLIP recall {g('rec')} FPR {g('fpr')}")
    L += ["", f"Crime type ({nc} classes, chance {100 / nc:.1f}, majority {100 * maj:.1f}):"]
    for k, rows in R["classes"].items():
        if k != "info":
            L.append(f"- {k}: acc {100 * np.mean([q['acc'] for q in rows]):.1f} | macro F1 {100 * np.mean([q['macro_f1'] for q in rows]):.1f}")
    open(os.path.join(OUT, "XD_SUMMARY.md"), "w").write("\n".join(L) + "\n")
    os.makedirs(os.path.join(WORK, "download"), exist_ok=True)
    zp = os.path.join(WORK, "download", "xd_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(OUT):
            z.write(os.path.join(OUT, f), os.path.join("xd_out", f))
    print("\n" + "\n".join(L))
    print(f"\nZIP: {zp} ({os.path.getsize(zp) / 1e6:.1f} MB) | total {(time.time() - T_START) / 3600:.2f} h")
    print("DONE | END OF REPORT")


if __name__ == "__main__":
    main()
