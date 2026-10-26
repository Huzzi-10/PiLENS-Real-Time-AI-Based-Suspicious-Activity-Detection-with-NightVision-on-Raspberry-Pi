"""PiLENS MoViNet Step 4 (Kaggle): ONNX export of the MoViNet backbone + binary head + crime-type head.
Run step 1 and step 2 (QUICK = False; needs binary_out/mlp_seed0.pt and norm.npz) first.
Output /kaggle/working/export_<TAG>/ (TAG = movinet_a0_v2 by default):
  backbone_<TAG>.onnx : clip (1,3,13,172,172) in [0,1] -> feature (1,2048)
  heads.onnx         : feat (1,2048) raw, recent (1,5,2048) raw -> binary_score (1,), class_logits (1,C)
                       (normalization mu/sd baked in; binary = MLP, crime type = mil_topk top-3)
  heads_meta.json    : classes, normalization, clip spec, recommended thresholds
Also prints a PyTorch vs ONNX Runtime parity check and Kaggle CPU latency (not the Pi number).
Then copy that folder to the Pi as export_movinet/.
"""
import os, sys, json, copy, time, subprocess, urllib.request
import numpy as np

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
def _find(default, name):
    """Pehle /kaggle/working, warna /kaggle/input mein saved notebook output (dataset folder skip)."""
    if os.path.exists(default):
        return default
    for dp, dn, fn in os.walk("/kaggle/input"):
        if "ufc-crime-full-dataset" in dp:
            dn[:] = []
            continue
        if name in dn or name in fn:
            return os.path.join(dp, name)
    return default

FEAT = _find(f"/kaggle/working/feats_{TAG}", f"feats_{TAG}")
BIN = f"/kaggle/working/binary_out_{TAG}"
OUT = f"/kaggle/working/export_{TAG}"
M_RECENT, TOPK, EPOCHS, FOLD = 5, 3, 80, 1
STAGE2 = "det"      # "det" = cascade protocol (13 anomaly classes), "fold" = purana 14-class (official fold 1)

setup_movinet()

import torch, torch.nn as nn, torch.nn.functional as F


class Backbone(nn.Module):
    """MoViNet -> 2048-d feature = conv7 + global pool + dense9 + Swish."""
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, x):
        m = s.m
        x = m.avg(m.conv7(m.blocks(m.conv1(x))))
        x = m.classifier[1](m.classifier[0](x))
        return x.flatten(1)


class Heads(nn.Module):
    def __init__(s, bin_net, fc14, mu, sd, pool="topk", T=1.0):
        super().__init__()
        s.bin, s.fc14, s.pool, s.T = bin_net, fc14, pool, float(T)
        s.register_buffer("mu", mu); s.register_buffer("sd", sd)
    def forward(s, feat, recent):
        score = torch.sigmoid(s.bin((feat - s.mu) / s.sd).squeeze(-1) / s.T)       # T>1: saturation kam (score 1.0 par chipakta nahi)
        z = s.fc14((recent - s.mu) / s.sd)
        logits = z.mean(1) if s.pool == "mean" else z.topk(TOPK, dim=1).values.mean(1)
        return score, logits


def macro_f1(y, p, n):
    cm = np.zeros((n, n), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    tp = np.diag(cm).astype(float)
    return float((2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)).mean())


def train_14class(mu, sd, dev):
    ar = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Action_Regnition_splits")
    classes = [l.split()[0] for l in open(os.path.join(ar, "ClassIDs.txt")) if l.strip()]
    cid = {c: i for i, c in enumerate(classes)}
    rd = lambda p: [(os.path.splitext(os.path.basename(l.split()[0]))[0], l.split()[0].split("/")[0]) for l in open(p) if l.strip()]
    tr_all, te = rd(os.path.join(ar, f"train_00{FOLD}.txt")), rd(os.path.join(ar, f"test_00{FOLD}.txt"))
    rs = np.random.RandomState(100 + FOLD); tr, va = [], []
    for c in classes:
        idx = [i for i, (_, k) in enumerate(tr_all) if k == c]; rs.shuffle(idx)
        nv = max(1, int(round(len(idx) * 0.15))); va += idx[:nv]; tr += idx[nv:]
    ld = lambda names: (torch.from_numpy(np.stack([np.load(os.path.join(FEAT, n + ".npy")).astype(np.float32) for n in names])) - mu) / sd
    Xtr, Xva, Xte = ld([tr_all[i][0] for i in tr]), ld([tr_all[i][0] for i in va]), ld([n for n, _ in te])
    ytr, yva = np.array([cid[tr_all[i][1]] for i in tr]), np.array([cid[tr_all[i][1]] for i in va])
    yte = np.array([cid[k] for _, k in te])
    torch.manual_seed(0); rng = np.random.RandomState(0)
    fc = nn.Linear(2048, len(classes)).to(dev); drop = nn.Dropout(0.5)
    opt = torch.optim.AdamW(fc.parameters(), lr=1e-3, weight_decay=1e-2)
    fwd = lambda x: fc(drop(x)).topk(TOPK, dim=1).values.mean(1)
    Xg, yg = Xtr.to(dev), torch.from_numpy(ytr).to(dev)
    best, state = -1, None
    for ep in range(EPOCHS):
        drop.train()
        o = rng.permutation(len(ytr))
        for b in range(0, len(o), 32):
            i = o[b:b + 32]; sub = np.sort(rng.choice(32, 16, replace=False))
            loss = F.cross_entropy(fwd(Xg[i][:, sub]), yg[i], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        drop.eval()
        with torch.no_grad():
            f1 = macro_f1(yva, fwd(Xva.to(dev)).argmax(1).cpu().numpy(), len(classes))
        if f1 > best:
            best, state = f1, copy.deepcopy(fc.state_dict())
    fc.load_state_dict(state); drop.eval()
    with torch.no_grad():
        acc = float((fwd(Xte.to(dev)).argmax(1).cpu().numpy() == yte).mean())
    print(f"14-class head (fold {FOLD}, mil_topk): val macro-F1 {best:.3f} | fold test acc {acc:.3f} (sanity, step3 jaisa ~0.3 hona chahiye)")
    return fc.cpu(), classes


def train_14class_det(mu, sd, dev):
    """Cascade protocol: 13 anomaly classes, Anomaly_Train ke anomaly videos par train, Anomaly_Test ke 140 anomaly videos par sanity.
    Normal sampling vs class-balanced sampling: chunav sirf val macro-F1 se (test se nahi)."""
    sp = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Anomaly_Detection_splits")
    rd = lambda p: [(os.path.splitext(os.path.basename(t[0]))[0], t[0].split("/")[0]) for t in (l.split() for l in open(p, errors="ignore")) if t and "Normal" not in t[0].split("/")[0]]
    tr_all, te = rd(os.path.join(ROOT, "Anomaly_Train.txt")), rd(os.path.join(sp, "Anomaly_Test.txt"))
    classes = sorted({k for _, k in tr_all}); cid = {c: i for i, c in enumerate(classes)}; nc = len(classes)
    assert nc == 13 and not ({n for n, _ in tr_all} & {n for n, _ in te}), "anomaly classes/overlap check fail"
    rs = np.random.RandomState(7); tr, va = [], []                      # step3 DET jaisa hi split
    for c in classes:
        idx = [i for i, (_, k) in enumerate(tr_all) if k == c]; rs.shuffle(idx)
        nv = max(1, int(round(len(idx) * 0.15))); va += idx[:nv]; tr += idx[nv:]
    ld = lambda names: (torch.from_numpy(np.stack([np.load(os.path.join(FEAT, n + ".npy")).astype(np.float32) for n in names])) - mu) / sd
    Xtr, Xva, Xte = ld([tr_all[i][0] for i in tr]), ld([tr_all[i][0] for i in va]), ld([n for n, _ in te])
    ytr, yva = np.array([cid[tr_all[i][1]] for i in tr]), np.array([cid[tr_all[i][1]] for i in va])
    yte = np.array([cid[k] for _, k in te])
    D = Xtr.shape[-1]
    fwd_k = lambda fc, drop, x, pool: (fc(drop(x)).mean(1) if pool == "mean" else fc(drop(x)).topk(TOPK, dim=1).values.mean(1))
    Xg, yg = Xtr.to(dev), torch.from_numpy(ytr).to(dev)
    w = 1.0 / np.bincount(ytr, minlength=nc)[ytr]; w /= w.sum()
    best_all = None
    for pool, wrs in (("mean", False), ("mean", True), ("topk", False), ("topk", True)):
        torch.manual_seed(0); rng = np.random.RandomState(0)
        fc = nn.Linear(D, nc).to(dev); drop = nn.Dropout(0.5)
        opt = torch.optim.AdamW(fc.parameters(), lr=1e-3, weight_decay=1e-2)
        best, state = -1, None
        for ep in range(EPOCHS):
            drop.train()
            o = rng.choice(len(ytr), len(ytr), p=w) if wrs else rng.permutation(len(ytr))
            for b in range(0, len(o), 32):
                i = o[b:b + 32]; sub = np.sort(rng.choice(32, 16, replace=False))
                loss = F.cross_entropy(fwd_k(fc, drop, Xg[i][:, sub], pool), yg[i], label_smoothing=0.1)
                opt.zero_grad(); loss.backward(); opt.step()
            drop.eval()
            with torch.no_grad():
                f1 = macro_f1(yva, fwd_k(fc, drop, Xva.to(dev), pool).argmax(1).cpu().numpy(), nc)
            if f1 > best:
                best, state = f1, copy.deepcopy(fc.state_dict())
        fc.load_state_dict(state); drop.eval()
        with torch.no_grad():
            acc = float((fwd_k(fc, drop, Xte.to(dev), pool).argmax(1).cpu().numpy() == yte).mean())
        print(f"stage-2 DET head (pool={pool}, wrs={wrs}): val macro-F1 {best:.3f} | 140 anomaly test videos acc {acc:.3f} (sirf report)")
        if best_all is None or best > best_all[0]:
            best_all = (best, (pool, wrs), fc.cpu())
    print(f"  -> val macro-F1 ke hisaab se chuna: pool={best_all[1][0]}, wrs={best_all[1][1]}")
    return best_all[2], classes, best_all[1][0]


def main():
    print("SCRIPT VERSION: step4 v3 (score temperature T, stage2 pool search)")
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nz = np.load(os.path.join(BIN, "norm.npz"))
    mu, sd = torch.from_numpy(nz["mu"]).float(), torch.from_numpy(nz["sd"]).float()

    # 1. Backbone
    mv = load_movinet()
    bb = Backbone(mv).eval()
    x = torch.rand(1, 3, T_FR, CROP_MV, CROP_MV)
    cap = {}
    h = mv.classifier[1].register_forward_hook(lambda mod, inp, out: cap.__setitem__("f", out.detach().flatten(1)))
    with torch.no_grad():
        f_wrap = bb(x)
        mv(x); h.remove()
    d = (f_wrap - cap["f"]).abs().max().item()
    print("backbone wrapper vs pretrained model hook: max diff", d, "| shape", tuple(f_wrap.shape))
    assert f_wrap.shape == (1, 2048) and d < 1e-4, "wrapper match nahi karta, ruk jao aur mujhe batao"
    bb_path = os.path.join(OUT, f"backbone_{TAG}.onnx")
    torch.onnx.export(bb, x, bb_path, input_names=["clip"], output_names=["feature"], opset_version=17, dynamo=False)

    # 2. Heads: binary MLP (step2) + 14-class (yahan train)
    bin_net = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.6), nn.Linear(512, 32), nn.ReLU(), nn.Dropout(0.6), nn.Linear(32, 1))
    bin_net.load_state_dict(torch.load(os.path.join(BIN, "mlp_seed0.pt"), map_location="cpu"))
    if STAGE2 == "det":
        fc14, classes, pool = train_14class_det(mu, sd, dev)
    else:
        (fc14, classes), pool = train_14class(mu, sd, dev), "topk"
    # score temperature T: val logits ke scale se (sirf agar logits bade hon; warna T=1). Pi threshold bhi yahin se.
    T, rec = 1.0, {}
    pk_path = os.path.join(BIN, "eval_pack.npz")
    if os.path.exists(pk_path):
        pk = np.load(pk_path)
        if "test_mlp_4" not in pk.files:
            raise SystemExit("ROKO: binary_out mein QUICK run ka model/pack hai (sirf 1 seed, 3 epoch). Pehle step2/m_step2 ko QUICK = False par poora chalao, phir ye script.")
        vs, vy = pk["val_mlp_0"], pk["val_y"]
        lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
        q = float(np.quantile(np.abs(lg(vs)), 0.995)); T = max(1.0, float(np.ceil(q / 8.0)))
        vm = vs.max(1); sg = lambda z: float(1 / (1 + np.exp(-z)))
        rec["youden"] = sg(lg(float(pk["thr_mlp_0"])) / T)
        for fp in (0.05, 0.10):
            rec[f"val_FPR_{int(fp * 100)}pct"] = sg(lg(float(np.quantile(vm[vy == 0], 1 - fp))) / T)
        print(f"score temperature T = {T:g} (val |logit| 99.5% = {q:.1f}) | Pi ke liye RECOMMENDED --thr: " + ", ".join(f"{k} {v:.4f}" for k, v in rec.items()))
    heads = Heads(bin_net, fc14, mu, sd, pool, T).eval()
    feat_d, rec_d = torch.randn(1, 2048), torch.randn(1, M_RECENT, 2048)
    hp = os.path.join(OUT, "heads.onnx")
    torch.onnx.export(heads, (feat_d, rec_d), hp, input_names=["feat", "recent"], output_names=["binary_score", "class_logits"], opset_version=17, dynamo=False)
    json.dump(dict(classes=classes, mu=mu.tolist(), sd=sd.tolist(), clip=dict(frames=T_FR, stride=STRIDE_MV, side=SIDE_MV, crop=CROP_MV, mean=MEAN_MV, std=STD_MV),
                   recent_clips=M_RECENT, topk=TOPK, pool=pool, stage2=STAGE2, score_T=T, thr_reco=rec, backbone=f"backbone_{TAG}.onnx", model=TAG, color="gray" if COLOR != "rgb" else "rgb"), open(os.path.join(OUT, "heads_meta.json"), "w"))
    for p in (bb_path, hp):
        print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.1f} MB")

    # 3. Parity: PyTorch vs ONNX Runtime, asli feature file par bhi
    import onnxruntime as ort
    sb = ort.InferenceSession(bb_path, providers=["CPUExecutionProvider"])
    sh = ort.InferenceSession(hp, providers=["CPUExecutionProvider"])
    o = sb.run(None, {"clip": x.numpy()})[0]
    print("backbone parity (torch vs ORT) max abs diff:", float(np.abs(o - f_wrap.numpy()).max()))
    real = np.load(os.path.join(FEAT, sorted(f for f in os.listdir(FEAT) if f.endswith(".npy"))[0])).astype(np.float32)
    feat_r, rec_r = real[:1], real[None, :M_RECENT]
    with torch.no_grad():
        ts, tl = heads(torch.from_numpy(feat_r), torch.from_numpy(rec_r))
    os_, ol = sh.run(None, {"feat": feat_r, "recent": rec_r})
    print("heads parity: score diff", float(np.abs(os_ - ts.numpy()).max()), "| logits diff", float(np.abs(ol - tl.numpy()).max()))
    print("sample: binary_score", float(os_[0]), "| top class", classes[int(ol.argmax())])

    # 4. Kaggle CPU latency (rough; Pi ka number nahi)
    for th in (1, 4):
        so = ort.SessionOptions(); so.intra_op_num_threads = th
        s = ort.InferenceSession(bb_path, so, providers=["CPUExecutionProvider"])
        for _ in range(3):
            s.run(None, {"clip": x.numpy()})
        ts_ = []
        for _ in range(15):
            t = time.perf_counter(); s.run(None, {"clip": x.numpy()}); ts_.append((time.perf_counter() - t) * 1000)
        print(f"Kaggle CPU, {th} thread: backbone median {np.median(ts_):.0f} ms, p95 {np.percentile(ts_, 95):.0f} ms")
    print("\nexport/ folder download karo: Kaggle Output tab se ya  !cd /kaggle/working && zip -r export.zip export")
    print("END OF REPORT")


main()
