"""PiLENS CNN baseline Step 2: 2D CNN (MobileNetV3-Large, frozen) + temporal model, weakly supervised MIL. Train + eval + ONNX export in one cell.

Input : /kaggle/working/feats_mnv3l/*.npy  (c_step1 output, (32, 13, 960) per video: 32 segments x 13 frames x 960)
Heads (on the 13 frame features of each clip):
  lstm        : CNN + LSTM (1 layer, hidden 256, last hidden state)
  transformer : CNN + Transformer (2 blocks, d 256, 4 heads, positional embedding)
  meanpool    : CNN + average over frames (no temporal model) = ablation of the temporal model
  Each has the same MIL head (256 -> 64 -> 1), the same loss (top-3 ranking hinge + smoothness + sparsity), the same split,
  the same val-chosen threshold and 5 seeds: exactly the X3D/MoViNet protocol, so the comparison is fair.
Output: /kaggle/working/binary_out_mnv3l/ (eval_pack.npz, test_meta.json, binary_results.json, norm.npz, *.pt)
        /kaggle/working/export_mnv3l/     (backbone_mnv3l.onnx, head_lstm.onnx, head_transformer.onnx, head_meanpool.onnx, heads_meta.json)
Then run step8_save_results.py to zip both folders.
"""
import os, sys, json, time, copy, math, subprocess
import numpy as np

QUICK = False        # False = poora run (paper ke liye). Sirf smoke test ke liye True
ROOT = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset"
FEAT = "/kaggle/working/feats_mnv3l"
META = "/kaggle/working/video_meta.csv"
OUT = "/kaggle/working/binary_out_mnv3l"
EXPORT = "/kaggle/working/export_mnv3l"
K, T_FR, D, E = 32, 13, 960, 256
SEEDS = [0] if QUICK else [0, 1, 2, 3, 4]
HEADS = ["lstm", "transformer", "meanpool"]
EPOCHS = 3 if QUICK else 40
BAG = 30
LR, WD, TOPK = 5e-4, 1e-4, 3
L_SMOOTH, L_SPARSE = 8e-5, 8e-5
VAL_FRAC, SPLIT_SEED = 0.15, 0
SMOOTH_SIGMA = 1.0
CLIP = dict(frames=13, stride=6, side=232, crop=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], per_frame=True)


# ---------------- metrics (numpy, ties sahi handle) ----------------
def auc_score(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    ranks = ((cum - cnt + 1) + cum) / 2.0
    r = ranks[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ap_score(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort"); s, y = s[o], y[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    ends = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp, fp = tp[ends], fp[ends]
    prec, rec = tp / (tp + fp), tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def gauss_smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(4 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    p = np.pad(x, (r, r), mode="reflect")
    return np.convolve(p, k, mode="valid")


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, (2 * p * r / max(p + r, 1e-9))


# ---------------- data ----------------
def read_list(p):
    out = []
    for l in open(p, errors="ignore"):
        l = l.strip().split()
        if l:
            path = l[0]
            out.append((os.path.splitext(os.path.basename(path))[0], 0 if "Normal" in path.split("/")[0] else 1))
    return out


def read_meta(p):
    """video_meta.csv -> {naam: frames}. Header ho ya na ho (resume/duplicate rows) dono chalte hain."""
    import csv
    out = {}
    if os.path.exists(p):
        for r in csv.reader(open(p)):
            if len(r) >= 3 and r[0] != "name":
                try:
                    out[os.path.splitext(r[0])[0]] = int(r[2])
                except ValueError:
                    pass
    return out


def fill_frames(frames, names):
    """Jo test video video_meta mein nahi, uske frames cv2 se seedha video se gino."""
    miss = {n for n in names if n not in frames}
    if not miss:
        return frames
    import cv2
    for dp, _, fs in os.walk(ROOT):
        for f in fs:
            n = os.path.splitext(f)[0]
            if n in miss and f.lower().endswith((".mp4", ".avi", ".mkv")):
                cap = cv2.VideoCapture(os.path.join(dp, f)); frames[n] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
                miss.discard(n)
    print(f"video_meta: kuch test videos ke frames video se gine{' | phir bhi nahi mile: ' + str(sorted(miss)[:5]) if miss else ''}")
    return frames


def read_ann(p):
    ann = {}
    for l in open(p, errors="ignore"):
        t = l.split()
        if len(t) >= 6:
            nm = os.path.splitext(t[0])[0]
            nums = [int(x) for x in t[2:6]]
            ann[nm] = [(nums[0], nums[1])] + ([(nums[2], nums[3])] if nums[2] >= 0 else []) if nums[0] >= 0 else []
    return ann


def frame_gt(n, segs):
    g = np.zeros(n, np.uint8)
    for s, e in segs:
        g[max(s, 0):min(e + 1, n)] = 1
    return g


def frame_scores(seg_scores, n):
    return seg_scores[np.minimum((np.arange(n) * K) // n, K - 1)]


# ---------------- model / train ----------------
def build(head):
    import torch, torch.nn as nn

    class Block(nn.Module):
        """Transformer block, haath se likha (ONNX export aur Pi par seedha chalta hai)."""
        def __init__(s, e=E, h=4, ff=512, p=0.1):
            super().__init__()
            s.h, s.ln1, s.ln2 = h, nn.LayerNorm(e), nn.LayerNorm(e)
            s.qkv, s.proj, s.drop = nn.Linear(e, 3 * e), nn.Linear(e, e), nn.Dropout(p)
            s.ff = nn.Sequential(nn.Linear(e, ff), nn.GELU(), nn.Linear(ff, e))
        def forward(s, x):
            t, e = x.shape[1], x.shape[2]; d = e // s.h
            q, k, v = s.qkv(s.ln1(x)).reshape(-1, t, 3, s.h, d).permute(2, 0, 3, 1, 4)
            att = ((q @ k.transpose(-2, -1)) * (d ** -0.5)).softmax(-1)
            y = (att @ v).transpose(1, 2).reshape(-1, t, e)
            x = x + s.drop(s.proj(y))
            return x + s.drop(s.ff(s.ln2(x)))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.kind, s.indrop = head, nn.Dropout(0.2)
            if head == "lstm":
                s.rnn = nn.LSTM(D, E, num_layers=1, batch_first=True)
            elif head == "transformer":
                s.inp, s.pos = nn.Linear(D, E), nn.Parameter(torch.randn(1, T_FR, E) * 0.02)
                s.blocks, s.ln = nn.Sequential(Block(), Block()), nn.LayerNorm(E)
            else:
                s.inp = nn.Sequential(nn.Linear(D, E), nn.ReLU())
            s.mlp = nn.Sequential(nn.Dropout(0.6), nn.Linear(E, 64), nn.ReLU(), nn.Dropout(0.6), nn.Linear(64, 1))
        def clip_embed(s, x):                      # x: N,T,D -> N,E
            x = s.indrop(x)
            if s.kind == "lstm":
                _, (h, _) = s.rnn(x)
                return h[-1]
            if s.kind == "transformer":
                return s.ln(s.blocks(s.inp(x) + s.pos)).mean(1)
            return s.inp(x.mean(1))
        def forward(s, x):                         # x: B,K,T,D -> B,K,1   ya   N,T,D -> N,1
            if x.dim() == 4:
                b, k = x.shape[0], x.shape[1]
                return s.mlp(s.clip_embed(x.reshape(b * k, x.shape[2], x.shape[3]))).reshape(b, k, 1)
            return s.mlp(s.clip_embed(x))
    return Net()


def seg_logits_of(model, X, dev):
    import torch
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            out.append(model(X[i:i + 32].to(dev).float()).squeeze(-1).double().cpu())
    return torch.cat(out).numpy()


def seg_scores_of(model, X, dev):
    """float64 sigmoid (float32 logit>17 par 1.0 ho jata hai)."""
    return 1.0 / (1.0 + np.exp(-seg_logits_of(model, X, dev)))


def train_one(head, seed, Xtr, ytr, Xva, yva, dev, topk=TOPK, ls=L_SMOOTH, lp=L_SPARSE):
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    m = build(head).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    ia, ino = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    best, best_state = -1, None
    for ep in range(EPOCHS):
        m.train()
        pa, pn = rng.permutation(ia), rng.permutation(ino)
        nb = min(len(pa), len(pn)) // BAG
        for b in range(nb):
            sa = torch.sigmoid(m(Xtr[pa[b * BAG:(b + 1) * BAG]].to(dev).float()).squeeze(-1))
            sn = torch.sigmoid(m(Xtr[pn[b * BAG:(b + 1) * BAG]].to(dev).float()).squeeze(-1))
            ta, tn = sa.topk(topk, 1).values.mean(1), sn.topk(topk, 1).values.mean(1)
            rank = F.relu(1 - ta[:, None] + tn[None, :]).mean()
            smooth = ((sa[:, 1:] - sa[:, :-1]) ** 2).sum(1).mean()
            loss = rank + ls * smooth + lp * sa.sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = auc_score(yva, seg_scores_of(m, Xva, dev).max(1))
        if v > best:
            best, best_state = v, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return m, best


# ---------------- ONNX export (Pi ke liye) ----------------
def export_all(models0, mu, sd, pack, yva, dev):
    import torch, torch.nn as nn
    for pkg in ("onnx", "onnxruntime"):
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
    import onnxruntime as ort
    import torchvision.models as tvm
    os.makedirs(EXPORT, exist_ok=True)

    class Backbone(nn.Module):
        def __init__(s):
            super().__init__()
            w = getattr(tvm.MobileNet_V3_Large_Weights, "IMAGENET1K_V2", tvm.MobileNet_V3_Large_Weights.DEFAULT)
            mm = tvm.mobilenet_v3_large(weights=w); s.features, s.pool = mm.features, mm.avgpool
        def forward(s, x):
            return s.pool(s.features(x)).flatten(1)

    bb = Backbone().eval()
    bp = os.path.join(EXPORT, "backbone_mnv3l.onnx")
    x = torch.randn(3, 3, 224, 224)
    torch.onnx.export(bb, x, bp, input_names=["frames"], output_names=["feature"], dynamic_axes={"frames": {0: "n"}, "feature": {0: "n"}},
                      opset_version=17, dynamo=False)
    diff = float(np.abs(ort.InferenceSession(bp, providers=["CPUExecutionProvider"]).run(None, {"frames": x.numpy()})[0] - bb(x).detach().numpy()).max())
    print(f"backbone_mnv3l.onnx {os.path.getsize(bp) / 1e6:.1f} MB | params {sum(p.numel() for p in bb.parameters()) / 1e6:.2f} M | ONNX vs torch max diff {diff:.2e}")

    class Head(nn.Module):
        def __init__(s, net, T):
            super().__init__(); s.net, s.T = net, float(T)
            s.register_buffer("mu", torch.from_numpy(mu).float()); s.register_buffer("sd", torch.from_numpy(sd).float())
        def forward(s, feats):                     # 1,13,960 raw MobileNet features -> score
            return torch.sigmoid(s.net((feats - s.mu) / s.sd).squeeze(-1) / s.T)

    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    meta = dict(model="mnv3l", backbone="backbone_mnv3l.onnx", feat_dim=D, clip=CLIP, heads={})
    for head, net in models0.items():
        net = net.cpu().eval()
        zv = np.log(pack[f"val_{head}_0"]) - np.log1p(-pack[f"val_{head}_0"])          # val segment logits
        zv = zv[np.isfinite(zv)]
        q = float(np.quantile(np.abs(zv), 0.995)); Tt = max(1.0, float(np.ceil(q / 8.0)))
        vmax = pack[f"val_{head}_0"].max(1); thr0 = float(pack[f"thr_{head}_0"])
        to_T = lambda p: float(sig(math.log(p / (1 - p)) / Tt)) if 0 < p < 1 else (1.0 if p >= 1 else 0.0)
        rec = {"youden": to_T(thr0), "val_fpr5": to_T(float(np.quantile(vmax[yva == 0], 0.95))), "val_fpr10": to_T(float(np.quantile(vmax[yva == 0], 0.90)))}
        h = Head(net, Tt).eval(); hp = os.path.join(EXPORT, f"head_{head}.onnx")
        f = torch.randn(1, T_FR, D) * torch.from_numpy(sd).float() + torch.from_numpy(mu).float()
        torch.onnx.export(h, f, hp, input_names=["feats"], output_names=["binary_score"], opset_version=17, dynamo=False)
        diff = float(np.abs(ort.InferenceSession(hp, providers=["CPUExecutionProvider"]).run(None, {"feats": f.numpy()})[0] - h(f).detach().numpy()).max())
        meta["heads"][head] = dict(file=f"head_{head}.onnx", score_T=Tt, thr_reco=rec, params=sum(p.numel() for p in net.parameters()))
        print(f"head_{head}.onnx {os.path.getsize(hp) / 1e6:.2f} MB | params {meta['heads'][head]['params'] / 1e6:.3f} M | T {Tt:g} | "
              f"Pi --thr: " + ", ".join(f"{k} {v:.4f}" for k, v in rec.items()) + f" | ONNX diff {diff:.2e}")
    json.dump(meta, open(os.path.join(EXPORT, "heads_meta.json"), "w"), indent=1)


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    print("SCRIPT VERSION: c_step2 v1 (CNN+LSTM / CNN-Transformer / CNN+meanpool, MIL, export) | MODE:", "QUICK (sirf test, paper ke liye nahi!)" if QUICK else "FULL")
    sp = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Anomaly_Detection_splits")
    train_all = read_list(os.path.join(ROOT, "Anomaly_Train.txt"))
    test = read_list(os.path.join(sp, "Anomaly_Test.txt"))
    frames = fill_frames(read_meta(META), [n for n, _ in test]); ann = read_ann(os.path.join(ROOT, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"))
    print(f"train {len(train_all)} (anomaly {sum(l for _, l in train_all)}) | test {len(test)} (anomaly {sum(l for _, l in test)}) | device {dev}")
    need = [n for n, _ in train_all + test]
    miss = [n for n in need if not os.path.exists(os.path.join(FEAT, n + ".npy"))]
    assert not miss, f"feature missing: {len(miss)} e.g. {miss[:3]} (pehle c_step1 poora chalao)"
    assert not ({n for n, _ in train_all} & {n for n, _ in test}), "train/test overlap!"

    rs = np.random.RandomState(SPLIT_SEED); tr, va = [], []          # same fixed split as X3D / MoViNet
    for lab in (0, 1):
        idx = [i for i, (_, l) in enumerate(train_all) if l == lab]; rs.shuffle(idx)
        nv = int(round(len(idx) * VAL_FRAC)); va += idx[:nv]; tr += idx[nv:]
    load = lambda names: np.stack([np.load(os.path.join(FEAT, n + ".npy")) for n in names])            # float16 N,K,T,D
    Xtr, Xva, Xte = load([train_all[i][0] for i in tr]), load([train_all[i][0] for i in va]), load([n for n, _ in test])
    ytr, yva = np.array([train_all[i][1] for i in tr]), np.array([train_all[i][1] for i in va])
    yte = np.array([l for _, l in test])
    acc1, acc2, cnt = np.zeros(D), np.zeros(D), 0
    for i in range(0, len(Xtr), 64):                                    # mean/std sirf train se, float64 mein
        c = Xtr[i:i + 64].reshape(-1, D).astype(np.float64); acc1 += c.sum(0); acc2 += (c ** 2).sum(0); cnt += len(c)
    mu = (acc1 / cnt).astype(np.float32); sd = (np.sqrt(np.maximum(acc2 / cnt - (acc1 / cnt) ** 2, 0)) + 1e-6).astype(np.float32)
    np.savez(os.path.join(OUT, "norm.npz"), mu=mu, sd=sd)
    nz = lambda X: torch.from_numpy(((X.astype(np.float32) - mu) / sd).astype(np.float16))
    Xtr, Xva, Xte = nz(Xtr), nz(Xva), nz(Xte)
    if dev == "cuda":
        Xtr, Xva, Xte = Xtr.to(dev), Xva.to(dev), Xte.to(dev)
    print(f"train {len(tr)} | val {len(va)} | test {len(test)} | features {tuple(Xtr.shape[1:])}")

    gts = [frame_gt(frames[n], ann.get(n, [])) for n, _ in test]
    gt_all = np.concatenate(gts)
    names_te = [n for n, _ in test]
    pack = dict(test_y=yte, val_y=yva, test_frames=np.array([frames[n] for n in names_te]))

    def evaluate(m, bval, tag):
        vs, ts = seg_scores_of(m, Xva, dev), seg_scores_of(m, Xte, dev)
        cand = np.unique(vs.max(1))
        thr = max(cand, key=lambda t: ((vs.max(1) >= t) & (yva == 1)).sum() / max((yva == 1).sum(), 1) - ((vs.max(1) >= t) & (yva == 0)).sum() / max((yva == 0).sum(), 1))
        p, r, f1 = prf(yte, (ts.max(1) >= thr).astype(int))
        row = dict(val_auc=bval, video_auc=auc_score(yte, ts.max(1)), thr=float(thr), prec=p, rec=r, f1=f1)
        for t2, sig in (("", 0.0), ("_smooth", SMOOTH_SIGMA)):
            fs = np.concatenate([frame_scores(gauss_smooth(ts[i], sig), frames[n]) for i, (n, _) in enumerate(test)])
            row["frame_auc" + t2] = auc_score(gt_all, fs); row["frame_ap" + t2] = ap_score(gt_all, fs)
        pack["val_" + tag], pack["test_" + tag], pack["thr_" + tag] = vs, ts, np.array(thr)
        return row

    results, models0 = {}, {}
    for head in HEADS:
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            m, bval = train_one(head, seed, Xtr, ytr, Xva, yva, dev)
            row = dict(seed=seed, **evaluate(m, bval, f"{head}_{seed}")); rows.append(row)
            print(f"[{head} seed {seed}] {time.time() - t0:.0f}s | " + " ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "seed"))
            torch.save(m.state_dict(), os.path.join(OUT, f"{head}_seed{seed}.pt"))
            if seed == 0:
                models0[head] = m
        results[head] = rows
        print(f"  {head} parameters: {sum(p.numel() for p in build(head).parameters()) / 1e6:.3f} M")

    np.savez_compressed(os.path.join(OUT, "eval_pack.npz"), **pack)
    json.dump(dict(names=names_te, y=yte.tolist(), ann={n: ann.get(n, []) for n in names_te}, seeds=SEEDS, heads=HEADS, smooth_sigma=SMOOTH_SIGMA,
                   train_n=len(tr), val_n=len(va), backbone="MobileNetV3-Large (ImageNet), per-frame 960-d"), open(os.path.join(OUT, "test_meta.json"), "w"))
    json.dump(results, open(os.path.join(OUT, "binary_results.json"), "w"), indent=1)

    print("\nSUMMARY (mean +- std over seeds; primary = unsmoothed)")
    for head, rows in results.items():
        ms = lambda k: f"{np.mean([r[k] for r in rows]):.3f} +- {np.std([r[k] for r in rows]):.3f}"
        print(f"{head:11s} video AUC {ms('video_auc')} | frame AUC {ms('frame_auc')} | frame AP {ms('frame_ap')} | P/R/F1 {ms('prec')}/{ms('rec')}/{ms('f1')}")

    print("\nONNX EXPORT (seed 0 models, Pi ke liye)")
    export_all(models0, mu, sd, pack, yva, dev)
    print("QUICK TEST OK" if QUICK else "DONE", "| END OF REPORT")


if __name__ == "__main__":
    main()
