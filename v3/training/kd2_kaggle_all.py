"""PiLENS KD2 (Kaggle, one cell, GPU T4, Internet ON): (1) A0 head variants, (2) cascade A0 alert -> CLIP confirm / crime type
in a short live-like window, (3) CLIP export for the Pi.
Add the output of the KD notebook (kd v1.0) as an input: it already holds the A0 and CLIP features, so nothing is extracted again
(without it the script still works, it just takes about 1.5 hours longer).
Output: download/kd2_results.zip (results) and download/export_clip_b16.zip (CLIP ONNX + heads, copy to the Pi as export_clip_b16/).
"""
import os, sys, json, time, copy, csv, subprocess, collections, urllib.request, zipfile, shutil
import numpy as np

SCRIPT_VERSION = "kd2 v1.2 (2026-10-13)"
QUICK = os.environ.get("PILENS_QUICK") == "1"          # sirf sandbox test ke liye; Kaggle par isay mat chhedo
ROOT = os.environ.get("PILENS_ROOT", "/kaggle/input/datasets/minmints/ufc-crime-full-dataset")
WORK = os.environ.get("PILENS_WORK", "/kaggle/working")
CLIP_NAME = "openai/clip-vit-base-patch16"
CLIP_RANDOM = os.environ.get("PILENS_CLIP_RANDOM") == "1"   # sandbox test (bina internet) - Kaggle par kabhi nahi
A0_TAG, CLIP_TAG = "movinet_a0_v2", "clip_b16"
OUT = os.path.join(WORK, "kd2_out")
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
LAM_SCORE, LAM_FEAT, LAM_CLS = 1.0, 1.0, 1.0      # pehle se fix
NFOLD = 2 if QUICK else 5
CBATCH, CLR, CWD = 32, 1e-3, 1e-2
NBOOT = 200 if QUICK else 2000
GROUPS = {"Violence": ["Abuse", "Assault", "Fighting", "Shooting", "Arrest"], "Theft": ["Burglary", "Robbery", "Shoplifting", "Stealing"],
          "Fire/damage": ["Arson", "Explosion", "Vandalism"], "Road accident": ["RoadAccidents"]}

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
            if "ufc-crime-full-dataset" in dp:
                dn[:] = []
                continue
            if name in dn or name in fn:
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

    meta_p = os.path.join(WORK, "video_meta_kd.csv")
    new = not os.path.exists(meta_p)
    mf = open(meta_p, "a", newline=""); mw = csv.writer(mf)
    if new:
        mw.writerow(["name", "path", "frames", "fps", "bad_reads"])
    dl = DataLoader(VC(todo), batch_size=None, num_workers=WORKERS, prefetch_factor=2 if WORKERS else None)
    t0, frames = time.time(), {}
    with torch.no_grad():
        for i, d in enumerate(dl, 1):
            nm = os.path.splitext(d["name"])[0]
            if d["a"] is None:
                print("PADHNE MEIN FAIL:", d["name"]); continue
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

    if head == "tattn":
        class TAttn(nn.Module):
            """Temporal attention head: Linear -> self-attention (4 heads, residual) -> per-segment score."""
            def __init__(s):
                super().__init__()
                s.inp = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.6))
                s.att = nn.MultiheadAttention(256, 4, dropout=0.3, batch_first=True)
                s.ln = nn.LayerNorm(256)
                s.out = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.6), nn.Linear(64, 1))
            def forward(s, x):
                h = s.inp(x); a, _ = s.att(h, h, h, need_weights=False)
                return s.out(s.ln(h + a))
        return TAttn()

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
    if head == "tattn":
        return None, m(x).squeeze(-1)
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


# =====================================================================================
# =====================================================================================
# KD2 main: behtar A0 head + cascade (A0 alert -> CLIP confirm / crime type, short window) + CLIP export (Pi)
# =====================================================================================
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


def onnx_export(model, x, path, **kw):
    """Purana (TorchScript) exporter: naye torch ke default exporter ko onnxscript chahiye, ye zaroori nahi."""
    import torch
    try:
        torch.onnx.export(model, x, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, x, path, **kw)


def main():
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    sp = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Anomaly_Detection_splits")
    train_all = [(n, 0 if "Normal" in c else 1, c) for n, c in read_list(os.path.join(ROOT, "Anomaly_Train.txt"))]
    test = [(n, 0 if "Normal" in c else 1, c) for n, c in read_list(os.path.join(sp, "Anomaly_Test.txt"))]
    assert not ({n for n, *_ in train_all} & {n for n, *_ in test}), "train/test overlap!"
    print(f"device {dev} | train {len(train_all)} (anomaly {sum(l for _, l, _ in train_all)}) | test {len(test)} (anomaly {sum(l for _, l, _ in test)})")
    assert 0 < sum(l for _, l, _ in train_all) < len(train_all), "labels galat"

    FA, FC, fr_new = extract_features({n for n, *_ in train_all + test})     # bane hue features (input se) skip hote hain
    has = lambda ds, n: any(os.path.exists(os.path.join(d, n + ".npy")) for d in ds)
    miss = [n for n, *_ in train_all + test if not (has(FA, n) and has(FC, n))]
    assert not miss, f"features missing {len(miss)}: {miss[:3]}"
    frames = read_meta([_find(os.path.join(WORK, "video_meta_kd.csv"), "video_meta_kd.csv"), _find(os.path.join(WORK, "video_meta.csv"), "video_meta.csv")])
    frames.update(fr_new)
    frames = fill_frames(frames, [n for n, *_ in test])
    ann = read_ann(os.path.join(ROOT, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"))
    assert all(n in ann for n, l, _ in test if l == 1), "annotation missing"

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
        D = Xtr.shape[-1]; mu, sd = Xtr.reshape(-1, D).mean(0), Xtr.reshape(-1, D).std(0) + 1e-6
        return [(x - mu) / sd for x in (Xtr, Xva, Xte)], (mu, sd)
    (Atr, Ava, Ate), normA = normed(FA)
    (Ctr, Cva, Cte), normC = normed(FC)
    print(f"train {len(tr)} | val {len(va)} | test {len(nte)} | A0 {tuple(Atr.shape)} | CLIP {tuple(Ctr.shape)}")

    U = [units(frames[n], ann.get(n, [])) for n in nte]
    UP, UN = np.stack([u[0] for u in U]), np.stack([u[1] for u in U])
    pack, R = dict(test_y=yte, val_y=yva, units_pos=UP, units_neg=UN), {"binary": {}, "bootstrap": {}, "cascade": {}, "cascade_classes": {}, "full_video_classes": {}}
    KEEP = {}

    frame_auc = lambda ts: wauc(ts.ravel(), UP.ravel(), UN.ravel())
    def frame_ap(ts):
        y = np.concatenate([np.r_[np.ones(int(p)), np.zeros(int(q))] for p, q in zip(UP.ravel(), UN.ravel())])
        s = np.concatenate([np.full(int(p + q), v) for p, q, v in zip(UP.ravel(), UN.ravel(), ts.ravel())])
        return ap_score(y, s)

    def run_rows(key, head, Xtr_, Xva_, Xte_):
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            m, bv = train_bin(head, seed, Xtr_, ytr, Xva_, yva, dev)
            vs, ts = seg_scores(m, Xva_, dev), seg_scores(m, Xte_, dev)
            cand = np.unique(vs.max(1))
            thr = max(cand, key=lambda t: ((vs.max(1) >= t) & (yva == 1)).sum() / max((yva == 1).sum(), 1) - ((vs.max(1) >= t) & (yva == 0)).sum() / max((yva == 0).sum(), 1))
            p, r, f1 = prf(yte, (ts.max(1) >= thr).astype(int))
            row = dict(seed=seed, val_auc=bv, video_auc=auc_score(yte, ts.max(1)), frame_auc=frame_auc(ts), frame_ap=frame_ap(ts), thr=float(thr), prec=p, rec=r, f1=f1,
                       normal_fpr=float(((ts.max(1) >= thr) & (yte == 0)).sum() / max((yte == 0).sum(), 1)))
            rows.append(row); pack[f"test_{key}_s{seed}"], pack[f"val_{key}_s{seed}"] = ts, vs
            if seed == 0:
                KEEP[key] = copy.deepcopy(m).cpu()
            print(f"  [{key} s{seed}] {time.time() - t0:.0f}s | val {bv:.3f} | video AUC {row['video_auc']:.3f} | frame AUC {row['frame_auc']:.4f} | AP {row['frame_ap']:.3f} | F1 {f1:.3f}", flush=True)
        R["binary"][key] = rows
        ms = lambda k: f"{np.mean([r[k] for r in rows]):.4f} +- {np.std([r[k] for r in rows]):.4f}"
        print(f"  == {key}: video AUC {ms('video_auc')} | frame AUC {ms('frame_auc')} | AP {ms('frame_ap')}")

    print("\n=== 1. BINARY: A0 heads (mlp, tconv, tattn = naya temporal-attention head) + CLIP mlp ===")
    for key, head, X in (("a0_mlp", "mlp", (Atr, Ava, Ate)), ("a0_tconv", "tconv", (Atr, Ava, Ate)), ("a0_tattn", "tattn", (Atr, Ava, Ate)), ("clip_mlp", "mlp", (Ctr, Cva, Cte))):
        run_rows(key, head, *X)

    print(f"\n=== 2. paired bootstrap ({NBOOT}x) ===")
    rng = np.random.RandomState(0)
    BS = [rng.randint(0, len(yte), len(yte)) for _ in range(NBOOT)]
    avg = lambda key: np.mean([pack[f"test_{key}_s{s}"] for s in SEEDS], 0)
    def boot(k1, k2):
        a, b = avg(k1), avg(k2); d_f, d_v = [], []
        for ix in BS:
            d_f.append(wauc(a[ix].ravel(), UP[ix].ravel(), UN[ix].ravel()) - wauc(b[ix].ravel(), UP[ix].ravel(), UN[ix].ravel()))
            d_v.append(auc_score(yte[ix], a[ix].max(1)) - auc_score(yte[ix], b[ix].max(1)))
        o = dict(frame_auc_diff=frame_auc(a) - frame_auc(b), frame_ci=[float(np.percentile(d_f, 2.5)), float(np.percentile(d_f, 97.5))],
                 video_auc_diff=auc_score(yte, a.max(1)) - auc_score(yte, b.max(1)), video_ci=[float(np.percentile(d_v, 2.5)), float(np.percentile(d_v, 97.5))])
        R["bootstrap"][f"{k1}__vs__{k2}"] = o
        print(f"  {k1} vs {k2}: frame {o['frame_auc_diff']:+.4f} [{o['frame_ci'][0]:+.4f}, {o['frame_ci'][1]:+.4f}] | video {o['video_auc_diff']:+.4f} [{o['video_ci'][0]:+.4f}, {o['video_ci'][1]:+.4f}]")
    boot("a0_tattn", "a0_tconv"); boot("a0_tattn", "a0_mlp"); boot("clip_mlp", "a0_mlp")

    print("\n=== 3. CASCADE: A0 alert -> CLIP confirm (A0 ke peak ke aas-paas +-r segment, live jaisa; mlp heads, har seed ka alag) ===")
    print("(primary: fpr5, r=0. Thresholds sirf val se. Test par ek baar.)")
    KINDS, RS = ("fpr5", "fpr10", "youden"), (0, 1, 2)
    casc = {(k, r): [] for k in KINDS for r in RS}
    alerts = {}
    for s in SEEDS:
        aV, aT = pack[f"val_a0_mlp_s{s}"], pack[f"test_a0_mlp_s{s}"]
        cV, cT = pack[f"val_clip_mlp_s{s}"], pack[f"test_clip_mlp_s{s}"]
        for k in KINDS:
            for r in RS:
                o, alertT = cascade_eval(aV, cV, yva, aT, cT, yte, k, r)
                casc[(k, r)].append(dict(seed=s, **o))
                if k == "fpr5":
                    alerts[(s, r)] = (alertT, aT.argmax(1))
    for (k, r), rows in casc.items():
        f = lambda x: f"{100 * np.mean([q[x] for q in rows]):.1f} +- {100 * np.std([q[x] for q in rows]):.1f}"
        R["cascade"][f"{k}_r{r}"] = rows
        print(f"  gate {k:7s} r={r} (window {2 * r + 1} seg) | A0 alone: recall {f('rec_a0')} FPR {f('fpr_a0')} | A0+CLIP: recall {f('rec')} FPR {f('fpr')}")

    print("\n=== 4. CRIME TYPE: A0 head vs CLIP head, cascade window (A0 ke alert par) ===")
    dtr_all = [(n, c) for n, l, c in train_all if l == 1]; dte = [(n, c) for n, l, c in test if l == 1]
    dcls = sorted({c for _, c in dtr_all}); dcid = {c: i for i, c in enumerate(dcls)}; dn = len(dcls)
    gid = {c: i for i, g in enumerate(GROUPS) for c in GROUPS[g]}
    rs = np.random.RandomState(7); dtr, dva = [], []
    for c in dcls:
        idx = [i for i, (_, k) in enumerate(dtr_all) if k == c]; rs.shuffle(idx)
        nv = max(1, int(round(len(idx) * VAL_FRAC))); dva += idx[:nv]; dtr += idx[nv:]
    def cnormed(d):
        Xtr, Xva, Xte = load(d, [dtr_all[i][0] for i in dtr]), load(d, [dtr_all[i][0] for i in dva]), load(d, [n for n, _ in dte])
        D = Xtr.shape[-1]; mu, sd = Xtr.reshape(-1, D).mean(0), Xtr.reshape(-1, D).std(0) + 1e-6
        return [(x - mu) / sd for x in (Xtr, Xva, Xte)], (mu, sd)
    CA, _ = cnormed(FA); CC, normCC = cnormed(FC)
    bidx = np.array([nte.index(n) for n, _ in dte])
    CLS_KEEP = {}
    for task, mapy, nc in (("classes", lambda c: dcid[c], dn), ("groups", lambda c: gid[c], 4)):
        ytr_c = np.array([mapy(dtr_all[i][1]) for i in dtr]); yva_c = np.array([mapy(dtr_all[i][1]) for i in dva]); yte_c = np.array([mapy(c) for _, c in dte])
        res = {f"{who}_{w}": [] for who in ("a0", "clip") for w in ("full", "cond_r0", "cond_r1", "cond_r2", "e2e_r0", "e2e_r1", "e2e_r2")}
        for s in SEEDS:
            for who, X in (("a0", CA), ("clip", CC)):
                lg, bv, mm = train_cls("mil_topk", s, X[0], ytr_c, X[1], yva_c, X[2], nc, dev)
                res[f"{who}_full"].append(float((lg.argmax(1).numpy() == yte_c).mean()))
                if s == 0 and who == "clip":
                    CLS_KEEP[task] = copy.deepcopy(mm).cpu()      # copy: asli model GPU par hi rahe (window eval ke liye)
                mm.eval()
                for r in RS:
                    alertT, pT = alerts[(s, r)]
                    ok = []
                    for j, i in enumerate(bidx):
                        if not alertT[i]:
                            continue
                        p = int(pT[i]); w = np.arange(max(p - r, 0), min(p + r + 1, K))     # window 32 segments ke andar
                        with torch.no_grad():
                            pr = mm(X[2][j:j + 1, w].to(dev)).argmax(1).item()
                        ok.append(pr == yte_c[j])
                    res[f"{who}_cond_r{r}"].append(float(np.mean(ok)) if ok else float("nan"))
                    res[f"{who}_e2e_r{r}"].append(float(np.sum(ok) / len(bidx)))
        R["cascade_classes"][task] = res
        f = lambda k: f"{100 * np.nanmean(res[k]):.1f} +- {100 * np.nanstd(res[k]):.1f}"
        print(f"\n--- {task} ({nc}) | chance {100 / nc:.1f} ---")
        print(f"  full video (32 segment, A0 ke alert ke bina): A0 {f('a0_full')} | CLIP {f('clip_full')}")
        for r in RS:
            print(f"  A0 alert (fpr5) par, window {2 * r + 1} seg: alert hui videos par accuracy A0 {f(f'a0_cond_r{r}')} | CLIP {f(f'clip_cond_r{r}')}"
                  f"   || end-to-end (miss = galat) A0 {f(f'a0_e2e_r{r}')} | CLIP {f(f'clip_e2e_r{r}')}")

    print("\n=== 5. EXPORT: CLIP ViT-B/16 (Pi ke liye) ===")
    EXP = os.path.join(WORK, "export_clip_b16"); os.makedirs(EXP, exist_ok=True)
    pip_need("onnx", "onnxruntime")
    import onnxruntime as ort
    pip_need("transformers")
    from transformers import CLIPVisionModelWithProjection, CLIPVisionConfig
    cm = (CLIPVisionModelWithProjection(CLIPVisionConfig(patch_size=16, projection_dim=512, num_hidden_layers=2)) if CLIP_RANDOM
          else CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME, torch_dtype=torch.float32)).eval().cpu()

    class ImgEnc(nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, x):
            return s.m(pixel_values=x).image_embeds
    xe = torch.randn(2, 3, CLIP_SIDE, CLIP_SIDE)
    p32 = os.path.join(EXP, "clip_b16_fp32.onnx")
    kw = dict(input_names=["x"], output_names=["e"], dynamic_axes={"x": {0: "n"}, "e": {0: "n"}}, opset_version=17)
    onnx_export(ImgEnc(cm), xe, p32, **kw)
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s32 = ort.InferenceSession(p32, so, providers=["CPUExecutionProvider"])
    xt = torch.randn(3, 3, CLIP_SIDE, CLIP_SIDE)
    with torch.no_grad():
        ref = ImgEnc(cm)(xt).numpy()
    o32 = s32.run(None, {"x": xt.numpy()})[0]
    print(f"  fp32 ONNX vs torch: max abs diff {np.abs(o32 - ref).max():.2e} | size {os.path.getsize(p32) / 1e6:.0f} MB")
    assert np.abs(o32 - ref).max() < 1e-2, "CLIP ONNX export galat"
    p8 = os.path.join(EXP, "clip_b16_int8.onnx")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(p32, p8, weight_type=QuantType.QInt8)
        s8 = ort.InferenceSession(p8, so, providers=["CPUExecutionProvider"])
        o8 = s8.run(None, {"x": xt.numpy()})[0]
        cosv = (o8 * o32).sum(1) / (np.linalg.norm(o8, axis=1) * np.linalg.norm(o32, axis=1))
        print(f"  int8 ONNX: size {os.path.getsize(p8) / 1e6:.0f} MB | cosine vs fp32 (random input, sirf ishara) {cosv.min():.3f}   (asli accuracy abhi check nahi: sirf speed ke liye)")
    except Exception as e:
        print("  int8 export fail (zaroori nahi):", repr(e)[:150])

    class HeadMLP(nn.Module):
        def __init__(s, m, mu, sd):
            super().__init__(); s.m = m; s.register_buffer("mu", mu); s.register_buffer("sd", sd)
        def forward(s, f):
            return s.m((f - s.mu) / s.sd)
    class HeadFC(nn.Module):
        def __init__(s, fc, mu, sd):
            super().__init__(); s.fc = fc; s.register_buffer("mu", mu); s.register_buffer("sd", sd)
        def forward(s, x):
            return s.fc((x - s.mu) / s.sd)
    mm = KEEP["clip_mlp"].eval()
    xf = torch.randn(5, 512)
    pb = os.path.join(EXP, "clip_head_binary.onnx")
    onnx_export(HeadMLP(mm, normC[0], normC[1]).eval(), xf, pb, input_names=["feat"], output_names=["logit"], dynamic_axes={"feat": {0: "n"}, "logit": {0: "n"}}, opset_version=17)
    with torch.no_grad():
        want = HeadMLP(mm, normC[0], normC[1]).eval()(xf).numpy()
    got = ort.InferenceSession(pb, providers=["CPUExecutionProvider"]).run(None, {"feat": xf.numpy()})[0]
    print(f"  binary head ONNX vs torch: max diff {np.abs(got - want).max():.2e}")
    assert np.abs(got - want).max() < 1e-4
    for task in ("classes", "groups"):
        hm = CLS_KEEP[task].eval(); pc = os.path.join(EXP, f"clip_head_{task}.onnx")
        xs = torch.randn(1, 4, 512)
        onnx_export(HeadFC(hm.fc, normCC[0], normCC[1]).eval(), xs, pc, input_names=["x"], output_names=["logits"], dynamic_axes={"x": {1: "s"}, "logits": {1: "s"}}, opset_version=17)
        lo = ort.InferenceSession(pc, providers=["CPUExecutionProvider"]).run(None, {"x": xs.numpy()})[0]            # 1,S,nc
        top = np.sort(lo, 1)[:, -min(TOPK, lo.shape[1]):].mean(1)
        with torch.no_grad():
            want = hm(((xs - normCC[0]) / normCC[1])).numpy()
        print(f"  {task} head ONNX (per-segment logits + top-{TOPK} mean) vs torch: max diff {np.abs(top - want).max():.2e}")
        assert np.abs(top - want).max() < 1e-4
    meta = dict(model="CLIP ViT-B/16 (openai/clip-vit-base-patch16) image encoder", input_size=CLIP_SIDE, rgb_mean=CLIP_MEAN.tolist(), rgb_std=CLIP_STD.tolist(),
                resize="short side 224, center crop 224, RGB", frames_of_13_clip=list(CLIP_PICK), embedding="image_embeds (512, not L2-normalised); avg of the 4 frames per clip",
                topk=TOPK, classes=dcls, groups=list(GROUPS), seed0_thresholds={f"{k}_r{r}": {kk: vv for kk, vv in casc[(k, r)][0].items() if kk.startswith("thr")} for k in KINDS for r in RS},
                note="thresholds are for the heads retrained in this script (A0 mlp seed0 + CLIP mlp seed0), raw sigmoid scores, no temperature")
    json.dump(meta, open(os.path.join(EXP, "heads_meta_clip.json"), "w"), indent=1)
    shutil.copy(pb, os.path.join(OUT, "clip_head_binary.onnx"))

    R["meta"] = dict(version=SCRIPT_VERSION, quick=QUICK, seeds=SEEDS, minutes=(time.time() - t_all) / 60)
    json.dump(R, open(os.path.join(OUT, "kd2_results.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "kd2_pack.npz"), **pack)

    L = [f"# PiLENS KD2 results ({SCRIPT_VERSION}, {'QUICK' if QUICK else 'FULL'})", "", "Official split, thresholds from val only, test once, 5 seeds.", "",
         "## Binary (A0 heads vs CLIP)", "", "| Model | Video AUC | Frame AUC | Frame AP | F1 | Normal FPR |", "|---|---|---|---|---|---|"]
    for k, rows in R["binary"].items():
        f = lambda x: f"{np.mean([r[x] for r in rows]):.3f} ± {np.std([r[x] for r in rows]):.3f}"
        L.append(f"| {k} | {f('video_auc')} | {f('frame_auc')} | {f('frame_ap')} | {f('f1')} | {f('normal_fpr')} |")
    L += ["", "## Paired bootstrap", "", "| Comparison | Frame AUC diff [95% CI] | Video AUC diff [95% CI] |", "|---|---|---|"]
    for k, o in R["bootstrap"].items():
        L.append(f"| {k.replace('__vs__', ' vs ')} | {o['frame_auc_diff']:+.4f} [{o['frame_ci'][0]:+.4f}, {o['frame_ci'][1]:+.4f}] | {o['video_auc_diff']:+.4f} [{o['video_ci'][0]:+.4f}, {o['video_ci'][1]:+.4f}] |")
    L += ["", "## Cascade: A0 alert -> CLIP confirm (window = +-r segments around A0 peak)", "", "| Gate | r | A0 recall | A0 FPR | Cascade recall | Cascade FPR |", "|---|---|---|---|---|---|"]
    for (k, r), rows in casc.items():
        f = lambda x: f"{100 * np.mean([q[x] for q in rows]):.1f} ± {100 * np.std([q[x] for q in rows]):.1f}"
        L.append(f"| {k} | {r} | {f('rec_a0')} | {f('fpr_a0')} | {f('rec')} | {f('fpr')} |")
    for task, res in R["cascade_classes"].items():
        f = lambda k: f"{100 * np.nanmean(res[k]):.1f} ± {100 * np.nanstd(res[k]):.1f}"
        L += ["", f"## Crime type: {task} (accuracy %, gate = A0 fpr5)", "", "| Window | A0 head cond. | CLIP head cond. | A0 end-to-end | CLIP end-to-end |", "|---|---|---|---|---|",
              f"| full video (no gate) | {f('a0_full')} | {f('clip_full')} | - | - |"]
        for r in RS:
            L.append(f"| {2 * r + 1} seg | {f(f'a0_cond_r{r}')} | {f(f'clip_cond_r{r}')} | {f(f'a0_e2e_r{r}')} | {f(f'clip_e2e_r{r}')} |")
    open(os.path.join(OUT, "KD2_SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    dl = os.path.join(WORK, "download"); os.makedirs(dl, exist_ok=True)
    z1 = os.path.join(dl, "kd2_results.zip")
    with zipfile.ZipFile(z1, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(OUT)):
            z.write(os.path.join(OUT, f), "kd2_out/" + f)
        z.write(os.path.join(EXP, "heads_meta_clip.json"), "heads_meta_clip.json")
    z2 = os.path.join(dl, "export_clip_b16.zip")
    with zipfile.ZipFile(z2, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in sorted(os.listdir(EXP)):
            z.write(os.path.join(EXP, f), "export_clip_b16/" + f)
    print(f"\nZIP 1 (chat mein bhejo): {z1} ({os.path.getsize(z1) / 1e6:.1f} MB)\nZIP 2 (Pi par le jao, chat mein nahi): {z2} ({os.path.getsize(z2) / 1e6:.0f} MB)")
    print(f"total {(time.time() - t_all) / 60:.0f} min")
    print("QUICK TEST OK" if QUICK else "DONE", "| END OF REPORT")


if __name__ == "__main__":
    main()
