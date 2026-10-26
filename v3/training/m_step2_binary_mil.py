"""PiLENS Step 2: binary (normal / suspicious) anomaly head, weakly supervised multiple-instance learning (MIL).
Input : /kaggle/working/feats_movinet_a0_v2/*.npy  (Step 1 output, (32, 2048) per video)
        /kaggle/working/video_meta.csv             (real frame counts)
Split : official Anomaly_Train.txt (1,610) and Anomaly_Test.txt (290).
        Validation = 15% of train, video level, one fixed split. No tuning on the test set.
Heads : MLP and TConv, 5 seeds. Loss = top-3 ranking hinge + smoothness + sparsity.
Metric: video AUC, frame AUC, frame AP (primary: unsmoothed) + a separate smoothed row.
        Frame ground truth uses each test video's real frame count (temporal annotation file).
Runs in one cell of a Kaggle GPU notebook. Set QUICK = True for a smoke test, then False.
For the paper (QUICK = False) it also writes:
  eval_pack.npz + test_meta.json : val and test scores, thresholds, labels, frame counts and annotations of every head/seed
                                   (evaluation/paper_eval.py builds CIs, ROC/PR, false alarms, alert rule, per-class results from these)
  ablation_results.json          : loss/top-k/head ablation (rank_only, topk1, topk5, linear_head), 3 seeds
The same protocol is used for every backbone (X3D-S, MoViNet).
"""
import os, sys, json, time, copy
import numpy as np

QUICK = False        # False = poora run (paper ke liye). Sirf 1-minute smoke test ke liye True karo
ROOT = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset"
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

VARIANT = os.environ.get("PILENS_VARIANT", "A0")                      # m_step1 jaisa hi rakho
TAG = "movinet_" + VARIANT.lower() + "_v2"      # v2 = pixels [0,1] (purani v1 (x-0.45)/0.225 galat thi)
TAG = os.environ.get("PILENS_TAG", TAG)          # night-vision run: movinet_a0_gray / movinet_a0_irsim
COLOR = os.environ.get("PILENS_COLOR", "rgb")     # rgb | gray | irsim (simulated night IR)
ALT = [t for t in os.environ.get("PILENS_ALT", "").split(",") if t]   # doosre test features (gray/irsim) par bhi isi model ka test
FEAT = _find(f"/kaggle/working/feats_{TAG}", f"feats_{TAG}")
META = _find("/kaggle/working/video_meta.csv", "video_meta.csv")
OUT = f"/kaggle/working/binary_out_{TAG}"
K = 32
SEEDS = [0] if QUICK else [0, 1, 2, 3, 4]
HEADS = ["mlp"] if QUICK else ["mlp", "tconv"]
EPOCHS = 3 if QUICK else 40
BAG = 30            # har step mein 30 anomaly + 30 normal video
LR, WD, TOPK = 5e-4, 1e-4, 3
L_SMOOTH, L_SPARSE = 8e-5, 8e-5
VAL_FRAC, SPLIT_SEED = 0.15, 0
SMOOTH_SIGMA = 1.0  # segments (pehle se fix)


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
def build(head, d=2048):
    import torch.nn as nn
    if head == "linear":
        return nn.Linear(d, 1)
    if head == "mlp":
        return nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.6), nn.Linear(512, 32), nn.ReLU(),
                             nn.Dropout(0.6), nn.Linear(32, 1))
    class TConv(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.Conv1d(d, 256, 3, padding=1), nn.ReLU(), nn.Dropout(0.6),
                                  nn.Conv1d(256, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.6), nn.Conv1d(64, 1, 1))
        def forward(s, x):
            return s.net(x.transpose(1, 2)).transpose(1, 2)
    return TConv()


def seg_scores_of(model, X, dev):
    """Segment scores = sigmoid(logit), lekin float64 mein: float32 sigmoid logit>17 par exactly 1.0 ho jata hai (ties, threshold fragile)."""
    import torch
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(model(X[i:i + 128].to(dev)).squeeze(-1).double().cpu())
    z = torch.cat(out).numpy()
    return 1.0 / (1.0 + np.exp(-z))


def train_one(head, seed, Xtr, ytr, Xva, yva, dev, topk=TOPK, ls=L_SMOOTH, lp=L_SPARSE):
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False   # TConv run-to-run farq kam karne ke liye
    m = build(head).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    ia, ino = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    Xg = Xtr.to(dev)
    best, best_state, hist = -1, None, []
    for ep in range(EPOCHS):
        m.train()
        pa, pn = rng.permutation(ia), rng.permutation(ino)
        nb = min(len(pa), len(pn)) // BAG
        tot = 0.0
        for b in range(nb):
            sa = torch.sigmoid(m(Xg[pa[b * BAG:(b + 1) * BAG]]).squeeze(-1))
            sn = torch.sigmoid(m(Xg[pn[b * BAG:(b + 1) * BAG]]).squeeze(-1))
            ta, tn = sa.topk(topk, 1).values.mean(1), sn.topk(topk, 1).values.mean(1)
            rank = F.relu(1 - ta[:, None] + tn[None, :]).mean()
            smooth = ((sa[:, 1:] - sa[:, :-1]) ** 2).sum(1).mean()
            sparse = sa.sum(1).mean()
            loss = rank + ls * smooth + lp * sparse
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        v = auc_score(yva, seg_scores_of(m, Xva, dev).max(1))
        hist.append((ep, tot / max(nb, 1), v))
        if v > best:
            best, best_state = v, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return m, best, hist


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    sp = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Anomaly_Detection_splits")
    train_all = read_list(os.path.join(ROOT, "Anomaly_Train.txt"))
    test = read_list(os.path.join(sp, "Anomaly_Test.txt"))
    frames = fill_frames(read_meta(META), [n for n, _ in test]); ann = read_ann(os.path.join(ROOT, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"))
    print("SCRIPT VERSION: step2 v3.1 (float64 scores, eval_pack, ablation, meta fix) | MODE:", "QUICK (sirf test, paper ke liye nahi!)" if QUICK else "FULL")
    print(f"train {len(train_all)} (anomaly {sum(l for _, l in train_all)}) | test {len(test)} (anomaly {sum(l for _, l in test)}) | device {dev}")
    need = [n for n, _ in train_all + test]
    miss = [n for n in need if not os.path.exists(os.path.join(FEAT, n + ".npy"))]
    assert not miss, f"feature missing: {len(miss)} e.g. {miss[:3]}"
    assert not ({n for n, _ in train_all} & {n for n, _ in test}), "train/test overlap!"
    missing_ann = [n for n, l in test if l == 1 and n not in ann]
    assert not missing_ann, f"annotation missing: {missing_ann[:3]}"

    # fixed video-level val split (stratified)
    rs = np.random.RandomState(SPLIT_SEED); tr, va = [], []
    for lab in (0, 1):
        idx = [i for i, (_, l) in enumerate(train_all) if l == lab]; rs.shuffle(idx)
        nv = int(round(len(idx) * VAL_FRAC)); va += idx[:nv]; tr += idx[nv:]
    load = lambda names: torch.from_numpy(np.stack([np.load(os.path.join(FEAT, n + ".npy")) for n in names]).astype(np.float32))
    Xtr, Xva, Xte = load([train_all[i][0] for i in tr]), load([train_all[i][0] for i in va]), load([n for n, _ in test])
    ytr, yva = np.array([train_all[i][1] for i in tr]), np.array([train_all[i][1] for i in va])
    yte = np.array([l for _, l in test])
    mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6   # sirf train se
    np.savez(os.path.join(OUT, "norm.npz"), mu=mu.numpy(), sd=sd.numpy())      # ONNX export / Pi ke liye zaroori
    Xtr, Xva, Xte = [(x - mu) / sd for x in (Xtr, Xva, Xte)]
    XALT = {}
    for t in ALT:
        fd = _find(f"/kaggle/working/feats_{t}", f"feats_{t}")
        XALT[t] = (torch.from_numpy(np.stack([np.load(os.path.join(fd, n + ".npy")) for n, _ in test]).astype(np.float32)) - mu) / sd
        print("extra test domain:", t, tuple(XALT[t].shape))
    print(f"train {len(tr)} | val {len(va)} | test {len(test)}")

    gts = [frame_gt(frames[n], ann.get(n, [])) for n, _ in test]
    gt_all = np.concatenate(gts)
    print(f"test frames {len(gt_all)}, anomaly frames {int(gt_all.sum())} ({100 * gt_all.mean():.1f}%)")

    names_te = [n for n, _ in test]
    pack = dict(test_y=yte, val_y=yva, test_frames=np.array([frames[n] for n in names_te]))

    def evaluate(m, bval, tag):
        """Val-chosen threshold (Youden), test par fix. Scores pack mein save."""
        vs, ts = seg_scores_of(m, Xva, dev), seg_scores_of(m, Xte, dev)
        cand = np.unique(vs.max(1))
        thr = max(cand, key=lambda t: ((vs.max(1) >= t) & (yva == 1)).sum() / max((yva == 1).sum(), 1) - ((vs.max(1) >= t) & (yva == 0)).sum() / max((yva == 0).sum(), 1))
        p, r, f1 = prf(yte, (ts.max(1) >= thr).astype(int))
        row = dict(val_auc=bval, video_auc=auc_score(yte, ts.max(1)), thr=float(thr), prec=p, rec=r, f1=f1)
        for t2, sig in (("", 0.0), ("_smooth", SMOOTH_SIGMA)):
            fs = np.concatenate([frame_scores(gauss_smooth(ts[i], sig), frames[n]) for i, (n, _) in enumerate(test)])
            row["frame_auc" + t2] = auc_score(gt_all, fs); row["frame_ap" + t2] = ap_score(gt_all, fs)
        pack["val_" + tag], pack["test_" + tag], pack["thr_" + tag] = vs, ts, np.array(thr)
        return row, ts

    results = {}
    for head in HEADS:
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            m, bval, hist = train_one(head, seed, Xtr, ytr, Xva, yva, dev)
            row, ts = evaluate(m, bval, f"{head}_{seed}"); row = dict(seed=seed, **row)
            for t, Xa in XALT.items():                 # wahi model, doosri qism ki test video (val threshold same)
                ta = seg_scores_of(m, Xa, dev); pack[f"xdom_{t}_{head}_{seed}"] = ta
                row["video_auc@" + t] = auc_score(yte, ta.max(1))
                row["frame_auc@" + t] = auc_score(gt_all, np.concatenate([frame_scores(ta[i], frames[n]) for i, (n, _) in enumerate(test)]))
                row["rec@" + t] = prf(yte, (ta.max(1) >= row["thr"]).astype(int))[1]
                row["fpr@" + t] = float(((ta.max(1) >= row["thr"]) & (yte == 0)).sum() / max((yte == 0).sum(), 1))
            rows.append(row)
            print(f"[{head} seed {seed}] {time.time() - t0:.0f}s | best val AUC {bval:.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in row.items() if k not in ("seed",)))
            torch.save(m.state_dict(), os.path.join(OUT, f"{head}_seed{seed}.pt"))
            np.save(os.path.join(OUT, f"test_scores_{head}_seed{seed}.npy"), ts)
        results[head] = rows
        print(f"  {head} head parameters: {sum(p.numel() for p in build(head).parameters()) / 1e6:.3f} M")

    # ---- ablation (paper): loss / top-k / head, mlp, 3 seeds ----
    abl = {}
    if not QUICK:
        print("\nABLATION (mlp-based, seeds 0-2; test par sirf report, selection val AUC se)")
        for name, kw in (("full", dict(head="mlp")), ("rank_only", dict(head="mlp", ls=0.0, lp=0.0)), ("topk1", dict(head="mlp", topk=1)),
                         ("topk5", dict(head="mlp", topk=5)), ("linear_head", dict(head="linear"))):
            rows = []
            for seed in (0, 1, 2):
                if name == "full" and "mlp" in results and seed in SEEDS:
                    rows.append(results["mlp"][SEEDS.index(seed)]); continue      # same run dobara nahi
                m, bval, _ = train_one(seed=seed, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, dev=dev, **kw)
                row, _ = evaluate(m, bval, f"abl_{name}_{seed}"); rows.append(dict(seed=seed, **row))
            abl[name] = rows
            ms = lambda k: f"{np.mean([r[k] for r in rows]):.3f} +- {np.std([r[k] for r in rows]):.3f}"
            print(f"  {name:12s} video AUC {ms('video_auc')} | frame AUC {ms('frame_auc')} | frame AP {ms('frame_ap')} | F1 {ms('f1')}")
        json.dump(abl, open(os.path.join(OUT, "ablation_results.json"), "w"), indent=1)

    np.savez_compressed(os.path.join(OUT, "eval_pack.npz"), **pack)
    json.dump(dict(names=names_te, y=yte.tolist(), ann={n: ann.get(n, []) for n in names_te}, seeds=SEEDS, heads=HEADS, smooth_sigma=SMOOTH_SIGMA,
                   train_n=len(tr), val_n=len(va)), open(os.path.join(OUT, "test_meta.json"), "w"))
    print(f"eval_pack.npz saved ({len(pack)} arrays)")

    print("\nSUMMARY (mean +- std over seeds; primary = unsmoothed)")
    for head, rows in results.items():
        ms = lambda k: f"{np.mean([r[k] for r in rows]):.3f} +- {np.std([r[k] for r in rows]):.3f}"
        print(f"{head:6s} video AUC {ms('video_auc')} | frame AUC {ms('frame_auc')} | frame AP {ms('frame_ap')} | "
              f"smooth frame AUC {ms('frame_auc_smooth')} | P/R/F1 {ms('prec')}/{ms('rec')}/{ms('f1')}")
    json.dump(results, open(os.path.join(OUT, "binary_results.json"), "w"), indent=1)
    print("QUICK TEST OK" if QUICK else "DONE", "| END OF REPORT")


if __name__ == "__main__":
    main()
