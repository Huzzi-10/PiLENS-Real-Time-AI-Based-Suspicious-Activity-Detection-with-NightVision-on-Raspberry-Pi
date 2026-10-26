"""PiLENS other-backbone experiments (ShuffleNet-3D, R(2+1)D-18). Official split, test used once. Runs in ONE Kaggle cell.

Each experiment has 3 stages (as in the main pipeline):
  STAGE 1  binary anomaly detection (video AUC, MIL top-k)
  STAGE 2  13 crime classes (crime videos only, 140 test videos)
  STAGE 3  4 groups (Violence / Theft / Fire-damage / Road accident)
Three ways to get the backbone:
  scratch   : trained from scratch (not pretrained) with the binary loss
  frozen    : pretrained weights, NOT trained at all, only features are extracted (the PiLENS approach)
  finetune  : pretrained weights, trained further with the binary loss
torchvision weights are Kinetics-400 (not K600) for r2plus1d_18, mc3_18 and s3d; MobileNetV3 uses ImageNet.
(MoViNet-A0 used by PiLENS is Kinetics-600; keep that difference in mind when comparing.)
The test set is used only once, at the end. Thresholds come from validation.

How to run: Kaggle notebook, add dataset minmints/ufc-crime-full-dataset, GPU T4, Internet ON, paste this whole file into one cell,
Save Version -> Save & Run All (Commit), then download download/*.zip from the Output tab.
"""
import os, sys, json, time, zipfile
import numpy as np

PERSON = "Warda"
EXPERIMENTS = [
 {
  "name": "shufflenet3d_scratch",
  "arch": "shufflenet3d",
  "mode": "scratch"
 },
 {
  "name": "r2plus1d18_k400_frozen",
  "arch": "r2plus1d_18",
  "mode": "frozen"
 },
 {
  "name": "r2plus1d18_k400_finetune",
  "arch": "r2plus1d_18",
  "mode": "finetune"
 }
]
QUICK = os.environ.get("QUICK") == "1"                 # 2-minute smoke test
ONLY = [int(i) for i in os.environ.get("ONLY", "").split(",") if i != ""]   # sirf kuch experiments (0,1,2)
ROOT = os.environ.get("ROOT", "/kaggle/input/datasets/minmints/ufc-crime-full-dataset")
W = os.environ.get("WORK", "/kaggle/working")
K, T, STRIDE, SIDE, BATCH = 8, 16, 2, 112, 4           # 8 segments x 16 frames (stride 2) x 112 px; batch = 4 videos (32 clips)
TOPK = 3
CACHE = os.path.join(W, f"cache_{SIDE}"); OUT = os.path.join(W, f"out_{PERSON.lower()}")
MEAN, STD = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)   # Kinetics stats (torchvision)
GROUPS = {"Violence": ["Abuse", "Assault", "Fighting", "Shooting", "Arrest"], "Theft": ["Burglary", "Robbery", "Shoplifting", "Stealing"],
          "Fire/damage": ["Arson", "Explosion", "Vandalism"], "Road accident": ["RoadAccidents"]}
OURS = "HAMARA (frozen MoViNet-A0, Kinetics-600): video AUC 0.919 | 13 classes 33% | 4 groups 59%   (X3D-S: 0.889 | 32% | 61%)"

import cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
cv2.setNumThreads(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- data (official split) ----------------
def read_list(p):
    return [(os.path.splitext(os.path.basename(t[0]))[0], t[0].split("/")[0]) for t in (l.split() for l in open(p, errors="ignore")) if t]


def find_videos():
    by = {}
    for dp, _, fs in sorted(os.walk(ROOT)):
        for f in sorted(fs):
            if f.lower().endswith((".mp4", ".avi", ".mkv")): by.setdefault(os.path.splitext(f)[0], os.path.join(dp, f))
    return by


class Decode(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        name, path = self.items[i]; dst = os.path.join(CACHE, name + ".npy")
        if os.path.exists(dst): return name
        cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); span = (T - 1) * STRIDE + 1
        clips, last = [], np.zeros((SIDE, SIDE, 3), np.uint8)
        for k in range(K):
            st = min(max(int((k + 0.5) * n / K) - span // 2, 0), max(n - span, 0)); cap.set(cv2.CAP_PROP_POS_FRAMES, st); fl = []
            for off in range(span):
                ok, fr = cap.read()
                if off % STRIDE: continue
                if ok:
                    h, w = fr.shape[:2]; s = SIDE / min(h, w); fr = cv2.resize(fr, (max(SIDE, round(w * s)), max(SIDE, round(h * s))))
                    h, w = fr.shape[:2]; y0, x0 = (h - SIDE) // 2, (w - SIDE) // 2
                    last = cv2.cvtColor(fr[y0:y0 + SIDE, x0:x0 + SIDE], cv2.COLOR_BGR2RGB)
                fl.append(last)
            clips.append(np.stack(fl))
        cap.release(); np.save(dst + ".tmp.npy", np.stack(clips)); os.replace(dst + ".tmp.npy", dst); return name


def build_cache(names, paths):
    os.makedirs(CACHE, exist_ok=True)
    todo = [(n, paths[n]) for n in names if not os.path.exists(os.path.join(CACHE, n + ".npy"))]
    print(f"cache: {len(names) - len(todo)} pehle se, {len(todo)} banani hain (~10 min)", flush=True); t0 = time.time()
    for i, _ in enumerate(DataLoader(Decode(todo), batch_size=None, num_workers=min(4, os.cpu_count() or 1)), 1):
        if i % 100 == 0 or i == len(todo): print(f"  {i}/{len(todo)} | {(time.time() - t0) / 60:.1f} min", flush=True)


class Clips(Dataset):
    def __init__(self, names, y, c, train): self.names, self.y, self.c, self.train = names, y, c, train
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        x = torch.from_numpy(np.load(os.path.join(CACHE, self.names[i] + ".npy")))      # K,T,H,W,3 uint8
        if self.train and np.random.rand() < 0.5: x = x.flip(3)
        return x, self.y[i], self.c[i]


def to_input(x, train):
    B, Kk, Tt, H, Wd, _ = x.shape
    x = x.float().div(255).permute(0, 1, 5, 2, 3, 4).reshape(B * Kk, 3, Tt, H, Wd)
    if train:
        g = x.new_empty(B, 1, 1, 1, 1).uniform_(0.7, 1.3).repeat_interleave(Kk, 0); b = x.new_empty(B, 1, 1, 1, 1).uniform_(-0.1, 0.1).repeat_interleave(Kk, 0)
        x = (x * g + b).clamp(0, 1)
        gray = (torch.rand(B, device=x.device) < 0.2).repeat_interleave(Kk)               # kabhi kabhi grayscale (raat ke liye)
        if gray.any(): x[gray] = x[gray].mean(1, keepdim=True).expand(-1, 3, -1, -1, -1)
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1, 1); s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1, 1)
    return (x - m) / s


# ---------------- backbones ----------------
def conv_bn(i, o, s): return nn.Sequential(nn.Conv3d(i, o, 3, s, 1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
def conv1(i, o): return nn.Sequential(nn.Conv3d(i, o, 1, 1, 0, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
def shuffle(x, g):
    b, c, d, h, w = x.size(); return x.view(b, g, c // g, d, h, w).transpose(1, 2).contiguous().view(b, -1, d, h, w)


class InvRes(nn.Module):
    def __init__(s, inp, oup, stride):
        super().__init__(); s.stride = stride; bf = oup // 2
        s.b1 = nn.Sequential(nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm3d(inp), nn.Conv3d(inp, bf, 1, 1, 0, bias=False),
                             nn.BatchNorm3d(bf), nn.ReLU(inplace=True)) if stride > 1 else nn.Sequential()
        s.b2 = nn.Sequential(nn.Conv3d(inp if stride > 1 else bf, bf, 1, 1, 0, bias=False), nn.BatchNorm3d(bf), nn.ReLU(inplace=True),
                             nn.Conv3d(bf, bf, 3, stride, 1, groups=bf, bias=False), nn.BatchNorm3d(bf), nn.Conv3d(bf, bf, 1, 1, 0, bias=False),
                             nn.BatchNorm3d(bf), nn.ReLU(inplace=True))
    def forward(s, x):
        if s.stride == 1: a, b = x.chunk(2, 1); o = torch.cat((a, s.b2(b)), 1)
        else: o = torch.cat((s.b1(x), s.b2(x)), 1)
        return shuffle(o, 2)


class ShuffleNet3D(nn.Module):                  # Warda ka 3D-ShuffleNetV2 (0.5x)
    out_dim = 1024
    def __init__(s):
        super().__init__(); ch = [24, 48, 96, 192, 1024]; s.conv1 = conv_bn(3, 24, (1, 2, 2)); s.pool = nn.MaxPool3d(3, 2, 1); L = []; ic = 24
        for si, rep in enumerate([4, 8, 4]):
            for i in range(rep): L.append(InvRes(ic, ch[si + 1], 2 if i == 0 else 1)); ic = ch[si + 1]
        s.features = nn.Sequential(*L); s.last = conv1(ic, 1024); s.gp = nn.AdaptiveAvgPool3d(1)
    def forward(s, x): return s.gp(s.last(s.features(s.pool(s.conv1(x))))).flatten(1)


class Block(nn.Module):
    def __init__(s, i, o, k=(3, 3, 3), st=1, p=1): super().__init__(); s.c = nn.Conv3d(i, o, k, st, p, bias=False); s.bn = nn.BatchNorm3d(o)
    def forward(s, x): return F.relu(s.bn(s.c(x)), inplace=True)


class SlowFastLight(nn.Module):                 # Fadia ka SlowFast-Light; fast = 16 frames, slow = har 4th frame
    out_dim = 320
    def __init__(s):
        super().__init__()
        s.fc1 = Block(3, 8, (5, 7, 7), (1, 2, 2), (2, 3, 3)); s.fr1 = Block(8, 16, st=(1, 2, 2)); s.fr2 = Block(16, 32, st=(1, 2, 2)); s.fr3 = Block(32, 64, st=(2, 2, 2))
        s.l1 = nn.Conv3d(8, 16, (5, 1, 1), (4, 1, 1), (2, 0, 0), bias=False); s.l2 = nn.Conv3d(16, 32, (5, 1, 1), (4, 1, 1), (2, 0, 0), bias=False)
        s.sc1 = Block(3, 32, (1, 7, 7), (1, 2, 2), (0, 3, 3)); s.sr1 = Block(48, 64, st=(1, 2, 2)); s.sr2 = Block(96, 128, st=(1, 2, 2)); s.sr3 = Block(128, 256, st=(2, 2, 2))
    def forward(s, x):
        fast, slow = x, x[:, :, ::4]
        f = s.fc1(fast); l1 = s.l1(f); f = s.fr1(f); l2 = s.l2(f); f = s.fr3(s.fr2(f)); ff = f.mean((2, 3, 4))
        z = s.sc1(slow); z = s.sr1(torch.cat([z, l1], 1)); z = s.sr3(s.sr2(torch.cat([z, l2], 1))); return torch.cat([z.mean((2, 3, 4)), ff], 1)


class TVBackbone(nn.Module):
    """torchvision video model (Kinetics-400 pretrained) -> feature vector."""
    def __init__(s, arch, pretrained):
        super().__init__(); from torchvision.models import video as V
        wcls = dict(r2plus1d_18="R2Plus1D_18_Weights", mc3_18="MC3_18_Weights", r3d_18="R3D_18_Weights", s3d="S3D_Weights")[arch]
        w = getattr(V, wcls).KINETICS400_V1 if (pretrained and not os.environ.get("NO_DOWNLOAD")) else None
        m = getattr(V, arch)(weights=w); s.s3d = arch == "s3d"
        if s.s3d: s.features, s.out_dim = m.features, 1024
        else: m.fc = nn.Identity(); s.m, s.out_dim = m, 512
    def forward(s, x): return s.features(x).mean((2, 3, 4)) if s.s3d else s.m(x)



class TSMNet(nn.Module):
    """MobileNetV3-Large + Temporal Shift (TSM): 2D CNN, har frame alag, shift se frames aapas mein baat karte hain. Pretrained = ImageNet (Kinetics nahi)."""
    out_dim = 960
    def __init__(s, pretrained):
        super().__init__(); from torchvision import models as M
        w = M.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if (pretrained and not os.environ.get("NO_DOWNLOAD")) else None
        m = M.mobilenet_v3_large(weights=w); s.nseg = T // 2
        s.features = m.features
        for i in range(2, len(s.features) - 1): s.features[i] = TShift(s.features[i], s.nseg)
        s.register_buffer("km", torch.tensor(MEAN).view(1, 3, 1, 1, 1)); s.register_buffer("ks", torch.tensor(STD).view(1, 3, 1, 1, 1))
        s.register_buffer("im", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)); s.register_buffer("istd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))
    def forward(s, x):
        x = (x * s.ks + s.km - s.im) / s.istd                       # Kinetics normalisation -> ImageNet normalisation
        x = x[:, :, ::2]; B, C, Tt, H, Wd = x.shape                # 8 frames
        f = s.features(x.permute(0, 2, 1, 3, 4).reshape(B * Tt, C, H, Wd))
        return f.mean((2, 3)).view(B, Tt, -1).mean(1)


class TShift(nn.Module):
    def __init__(s, net, nseg, div=8): super().__init__(); s.net, s.nseg, s.div = net, nseg, div
    def forward(s, x):
        bt, c, h, w = x.shape; x = x.view(bt // s.nseg, s.nseg, c, h, w); f = c // s.div; o = torch.zeros_like(x)
        o[:, :-1, :f] = x[:, 1:, :f]; o[:, 1:, f:2 * f] = x[:, :-1, f:2 * f]; o[:, :, 2 * f:] = x[:, :, 2 * f:]
        return s.net(o.view(bt, c, h, w))

def make_backbone(arch, pretrained):
    if arch == "mnv3_tsm": return TSMNet(pretrained)
    if arch == "shufflenet3d": return ShuffleNet3D()
    if arch == "slowfast_light": return SlowFastLight()
    return TVBackbone(arch, pretrained)


# ---------------- metrics ----------------
def auc(y, s):
    from scipy.stats import rankdata
    y, s = np.asarray(y), np.asarray(s); pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = rankdata(np.concatenate([pos, neg])); return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum(); f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def youden(vs, yv):
    return max(np.unique(vs), key=lambda t: ((vs >= t) & (yv == 1)).sum() / max((yv == 1).sum(), 1) - ((vs >= t) & (yv == 0)).sum() / max((yv == 0).sum(), 1))


# ---------------- STAGE 0: backbone (scratch / frozen / finetune) ----------------
def train_backbone(bb, mode, dl_tr, dl_va, epochs, ck):
    d = bb.out_dim; seg = nn.Linear(d, 1).to(DEV); drop = nn.Dropout(0.4)
    lr = 3e-4 if mode == "scratch" else 5e-5
    opt = torch.optim.AdamW([dict(params=bb.parameters(), lr=lr), dict(params=seg.parameters(), lr=1e-3)], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, 1e-3], total_steps=epochs * len(dl_tr), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda"); best = -1; t0 = time.time()
    def score(x):
        f = drop(bb(x)); return seg(f).view(-1, K)
    for ep in range(epochs):
        bb.train(); tl, n = 0.0, 0
        for x, y, c in dl_tr:
            x, y = x.to(DEV), y.to(DEV).float()
            with torch.autocast("cuda", enabled=DEV == "cuda"): z = score(to_input(x, True))
            loss = F.binary_cross_entropy_with_logits(z.float().topk(min(TOPK, K), 1).values.mean(1), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item() * len(y); n += len(y)
        bb.eval(); drop.eval(); S, Y = [], []
        with torch.no_grad():
            for x, y, c in dl_va:
                with torch.autocast("cuda", enabled=DEV == "cuda"): S.append(score(to_input(x.to(DEV), False)).float().cpu()); Y.append(y)
        va = auc(torch.cat(Y).numpy(), torch.cat(S).numpy().max(1))
        if va > best: best = va; torch.save(bb.state_dict(), ck)
        print(f"  epoch {ep + 1:02d}/{epochs} | train loss {tl / n:.3f} | val video AUC {va:.3f} | best {best:.3f} | {(time.time() - t0) / 60:.1f} min", flush=True)
    bb.load_state_dict(torch.load(ck, map_location=DEV)); return best


@torch.no_grad()
def extract(bb, dl):
    bb.eval(); Fe = []
    for x, y, c in dl:
        with torch.autocast("cuda", enabled=DEV == "cuda"): f = bb(to_input(x.to(DEV), False))
        Fe.append(f.float().view(-1, K, f.shape[-1]).cpu())
    return torch.cat(Fe)


# ---------------- STAGES 1-3: heads on features ----------------
def fit_binary(Xtr, ytr, Xva, yva, Xte, seed, epochs):
    torch.manual_seed(seed); rs = np.random.RandomState(seed); D = Xtr.shape[-1]
    net = nn.Sequential(nn.Linear(D, 512), nn.ReLU(), nn.Dropout(0.6), nn.Linear(512, 32), nn.ReLU(), nn.Dropout(0.6), nn.Linear(32, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); Xg, yg = Xtr.to(DEV), torch.tensor(ytr).float().to(DEV); best, bs = -1, None
    seg = lambda X: net(X.to(DEV)).squeeze(-1)
    for ep in range(epochs):
        net.train(); order = rs.permutation(len(ytr))
        for b in range(0, len(order), 32):
            i = order[b:b + 32]; z = net(Xg[i]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(z.topk(min(TOPK, K), 1).values.mean(1), yg[i]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = auc(yva, seg(Xva).double().cpu().numpy().max(1))
        if v > best: best, bs = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(bs); net.eval()
    with torch.no_grad(): return best, seg(Xva).double().cpu().numpy(), seg(Xte).double().cpu().numpy()


def fit_class(Xtr, ctr, Xva, cva, Xte, nc, seed, epochs):
    torch.manual_seed(seed); rs = np.random.RandomState(seed); D = Xtr.shape[-1]
    net = nn.Sequential(nn.Dropout(0.5), nn.Linear(D, nc)).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    cnt = np.bincount(ctr, minlength=nc).astype(np.float32); w = torch.tensor((1 / np.maximum(cnt, 1)) / (1 / np.maximum(cnt, 1)).mean()).to(DEV)
    Xg, yg = Xtr.to(DEV), torch.tensor(ctr).to(DEV); best, bs = -1, None
    pool = lambda X: net(X.to(DEV)).topk(min(TOPK, K), 1).values.mean(1)
    for ep in range(epochs):
        net.train(); order = rs.permutation(len(ctr))
        for b in range(0, len(order), 32):
            i = order[b:b + 32]; loss = F.cross_entropy(net(Xg[i]).topk(min(TOPK, K), 1).values.mean(1), yg[i], weight=w, label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): f1 = macro_f1(cva, pool(Xva).argmax(1).cpu().numpy(), nc)
        if f1 > best: best, bs = f1, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(bs); net.eval()
    with torch.no_grad(): return best, pool(Xte).cpu().numpy()


def run_experiment(idx, exp, D):
    name, arch, mode = exp["name"], exp["arch"], exp["mode"]; od = os.path.join(OUT, f"{idx}_{name}"); os.makedirs(od, exist_ok=True)
    print(f"\n{'=' * 72}\nEXPERIMENT {idx}: {name} | arch {arch} | mode {mode}\n{'=' * 72}", flush=True)
    bb = make_backbone(arch, mode != "scratch").to(DEV); npar = sum(p.numel() for p in bb.parameters()) / 1e6
    print(f"backbone params {npar:.2f} M | pretrained {('ImageNet-1k (torchvision)' if arch == 'mnv3_tsm' else 'Kinetics-400 (torchvision)') if mode != 'scratch' else 'nahi (scratch)'}", flush=True)
    t0 = time.time(); bval = None
    if mode in ("scratch", "finetune"):
        ep = (2 if QUICK else (25 if mode == "scratch" else 8)); print(f"STAGE 0: backbone binary loss se train ({ep} epochs)", flush=True)
        dl = D["dl_tr"] if mode == "scratch" else D["mk_tr"](2)       # finetune: chhota batch (2 videos) taake T4 ki memory kafi rahe
        bval = train_backbone(bb, mode, dl, D["dl_va"], ep, os.path.join(od, "backbone_best.pt"))
    else:
        for p in bb.parameters(): p.requires_grad = False
        print("STAGE 0: backbone frozen (train nahi hota)", flush=True)
    F_tr, F_va, F_te = extract(bb, D["dl_tr_e"]), extract(bb, D["dl_va"]), extract(bb, D["dl_te"])
    mu, sd = F_tr.reshape(-1, F_tr.shape[-1]).mean(0), F_tr.reshape(-1, F_tr.shape[-1]).std(0) + 1e-6
    F_tr, F_va, F_te = [(f - mu) / sd for f in (F_tr, F_va, F_te)]
    print(f"features: train {tuple(F_tr.shape)} val {tuple(F_va.shape)} test {tuple(F_te.shape)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    seeds = [0] if QUICK else [0, 1, 2, 3, 4]; res = dict(name=name, arch=arch, mode=mode, params_M=npar, backbone_val_auc=bval)
    # STAGE 1: binary
    print("\nSTAGE 1: binary anomaly detection", flush=True); rows = []
    for s in seeds:
        va, sv, st = fit_binary(F_tr, D["ytr"], F_va, D["yva"], F_te, s, 3 if QUICK else 40)
        vs, ts = sv.max(1), st.max(1); thr = youden(vs, D["yva"])
        rows.append(dict(seed=s, val_auc=va, video_auc=auc(D["yte"], ts), recall=float(((ts >= thr) & (D["yte"] == 1)).sum() / max((D["yte"] == 1).sum(), 1)),
                         fpr=float(((ts >= thr) & (D["yte"] == 0)).sum() / max((D["yte"] == 0).sum(), 1))))
        print(f"  seed {s}: val AUC {va:.3f} | TEST video AUC {rows[-1]['video_auc']:.3f} | recall {rows[-1]['recall']:.2f} | false alarm {rows[-1]['fpr']:.2f}", flush=True)
    res["binary"] = rows
    # STAGE 2: 13 classes ; STAGE 3: 4 groups (sirf crime videos)
    mtr, mva, mte = D["ctr"] >= 0, D["cva"] >= 0, D["cte"] >= 0
    for stage, key, nc, ytr_, yva_, yte_ in (("STAGE 2: 13 crime classes", "classes13", 13, D["ctr"][mtr], D["cva"][mva], D["cte"][mte]),
                                           ("STAGE 3: 4 groups", "groups4", 4, D["gtr"][mtr], D["gva"][mva], D["gte"][mte])):
        print("\n" + stage + f" (test {int(mte.sum())} crime videos)", flush=True); rows = []
        for s in seeds:
            f1v, lg = fit_class(F_tr[mtr], ytr_, F_va[mva], yva_, F_te[mte], nc, s, 5 if QUICK else 60); pred = lg.argmax(1)
            rows.append(dict(seed=s, val_f1=f1v, acc=float((pred == yte_).mean()), macro_f1=macro_f1(yte_, pred, nc),
                             top3=float(np.mean([yte_[i] in np.argsort(-lg[i])[:3] for i in range(len(yte_))])) if nc > 3 else 1.0))
            print(f"  seed {s}: val F1 {f1v:.3f} | TEST acc {rows[-1]['acc']:.3f} | macro-F1 {rows[-1]['macro_f1']:.3f}", flush=True)
            if s == seeds[0]:
                cm = np.zeros((nc, nc), int)
                for a, b in zip(yte_, pred): cm[a, b] += 1
                np.savetxt(os.path.join(od, f"confusion_{key}.csv"), cm, fmt="%d", delimiter=",")
        res[key] = rows
    json.dump(res, open(os.path.join(od, "results.json"), "w"), indent=1)
    return res


def summary(all_res):
    ms = lambda rows, k, sc=100: f"{sc * np.mean([r[k] for r in rows]):.1f}+-{sc * np.std([r[k] for r in rows]):.1f}"
    L = [f"RESULT {PERSON} (official split, test sirf ek dafa; binary/13/4 groups)", "",
         "| experiment | pretrained | params | video AUC | recall / false-alarm | 13 classes acc (chance 7.7) | top-3 | 4 groups acc (chance 31.4) |", "|---|---|---|---|---|---|---|---|"]
    for r in all_res:
        ds_ = "ImageNet" if r["arch"] == "mnv3_tsm" else "Kinetics-400"
        pre = "scratch" if r["mode"] == "scratch" else (f"{ds_} frozen" if r["mode"] == "frozen" else f"{ds_} fine-tuned")
        b = r["binary"]; L.append(f"| {r['name']} | {pre} | {r['params_M']:.1f}M | {np.mean([x['video_auc'] for x in b]):.3f}+-{np.std([x['video_auc'] for x in b]):.3f} | "
                                  f"{ms(b, 'recall')}% / {ms(b, 'fpr')}% | {ms(r['classes13'], 'acc')}% | {ms(r['classes13'], 'top3')}% | {ms(r['groups4'], 'acc')}% |")
    L += ["", OURS]; return "\n".join(L)


def main():
    print(f"SCRIPT VERSION: {PERSON.lower()}_kaggle v2.0 (2026-10-12) | MODE: {'QUICK (sirf test)' if QUICK else 'FULL'} | device {DEV}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    sp = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Anomaly_Detection_splits")
    tr_all, te = read_list(os.path.join(ROOT, "Anomaly_Train.txt")), read_list(os.path.join(sp, "Anomaly_Test.txt"))
    assert not ({n for n, _ in tr_all} & {n for n, _ in te}), "train/test overlap!"
    crimes = sorted({k for _, k in tr_all + te if "Normal" not in k}); assert len(crimes) == 13, crimes
    cid = {c: i for i, c in enumerate(crimes)}; gid = {c: i for i, g in enumerate(GROUPS) for c in GROUPS[g]}
    lab = lambda L: (np.array([0 if "Normal" in k else 1 for _, k in L]), np.array([-1 if "Normal" in k else cid[k] for _, k in L]),
                     np.array([-1 if "Normal" in k else gid[k] for _, k in L]))
    paths = find_videos()
    if QUICK:
        r0 = np.random.RandomState(0); tr_all = [tr_all[i] for i in r0.permutation(len(tr_all))[:60]]; te = [te[i] for i in r0.permutation(len(te))[:30]]
    miss = [n for n, _ in tr_all + te if n not in paths]; assert not miss, f"video nahi mili: {miss[:3]}"
    print(f"train {len(tr_all)} | test {len(te)} (official split)", flush=True); build_cache([n for n, _ in tr_all + te], paths)
    rs = np.random.RandomState(7); tr, va = [], []
    for k in sorted({k for _, k in tr_all}):
        idx = [i for i, (_, kk) in enumerate(tr_all) if kk == k]; rs.shuffle(idx); nv = max(1, int(round(len(idx) * 0.15))) if len(idx) > 1 else 0; va += idx[:nv]; tr += idx[nv:]
    sub = lambda ids: [tr_all[i] for i in ids]
    (ytr, ctr, gtr), (yva, cva, gva), (yte, cte, gte) = lab(sub(tr)), lab(sub(va)), lab(te)
    ds = lambda L, y, c, t: Clips([n for n, _ in L], torch.tensor(y), torch.tensor(c), t)
    mk = lambda d, **kw: DataLoader(d, batch_size=BATCH, num_workers=min(4, os.cpu_count() or 1), pin_memory=True, **kw)
    wt = np.where(ytr == 1, 0.5 / max((ytr == 1).sum(), 1), 0.5 / max((ytr == 0).sum(), 1))
    mk_tr = lambda b: DataLoader(ds(sub(tr), ytr, ctr, True), batch_size=b, num_workers=min(4, os.cpu_count() or 1), pin_memory=True,
                                 sampler=WeightedRandomSampler(wt, len(tr), replacement=True))
    D = dict(mk_tr=mk_tr, dl_tr=mk_tr(BATCH), dl_tr_e=mk(ds(sub(tr), ytr, ctr, False)),
             dl_va=mk(ds(sub(va), yva, cva, False)), dl_te=mk(ds(te, yte, cte, False)),
             ytr=ytr, yva=yva, yte=yte, ctr=ctr, cva=cva, cte=cte, gtr=gtr, gva=gva, gte=gte)
    all_res = []
    for i, exp in enumerate(EXPERIMENTS):
        if ONLY and i not in ONLY: continue
        try: all_res.append(run_experiment(i, exp, D))
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"!!! FAIL: experiment {i} {exp['name']}: {e}\nBaaqi experiments chalte rahenge.", flush=True)
        if all_res:
            rep = summary(all_res); open(os.path.join(OUT, "report.md"), "w").write(rep + "\n")
    if not all_res: raise SystemExit("ROKO: koi experiment pura nahi hua; upar wala error bhejo.")
    print("\n" + summary(all_res) + "\nEND OF REPORT", flush=True)
    os.makedirs(os.path.join(W, "download"), exist_ok=True); zp = os.path.join(W, "download", f"{PERSON.lower()}_results.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(OUT):
            for f in fs: z.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), OUT))
    print(f"zip: {zp} ({os.path.getsize(zp) / 1e6:.1f} MB) -> ye file Huzaifa ko do")


main()
