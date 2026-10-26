"""PiLENS Step 3: crime-type recognition head (stage 2) on frozen backbone features.
Input : /kaggle/working/feats_movinet_a0_v2/*.npy (Step 1), official UCF-Crime 4-fold split
        (Action_Regnition_splits/train_00k.txt, test_00k.txt: 532 train / 168 test, 38/12 per class).
Val   : 15% of each fold's train, video level, per class (used only for epoch selection).
Heads : linear (mean of segments) and mil_topk (per-segment logits, top-3 mean). 4 folds x 3 seeds.
Report: accuracy, macro-F1, top-3 accuracy (mean +- std over folds), then pooled per-class
        precision/recall/F1 and confusion matrix (seed 0, test videos of all four folds).
USE_WRS = True adds class-balanced sampling (inverse class frequency) for imbalanced splits.
Set QUICK = True for a smoke test, then False.
Cascade protocol ("DET"): stage 2 runs only on alerts, so it is trained on the 13 anomaly classes of the
Anomaly_Train anomaly videos (810) and tested on the 140 Anomaly_Test anomaly videos (the same test videos as the
binary head, so the cascade can be evaluated end to end). Linear / mil_topk x (normal / class-balanced "wrs"), 3 seeds.
Output: cls14_pack.npz (all logits) + cls14_meta.json (names/labels), confusion_*.csv, cls14_results.json.
"""
import os, json, time, copy
import numpy as np

QUICK = False        # False = poora run (paper ke liye). Sirf 1-minute smoke test ke liye True karo
USE_WRS = False
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
FEAT = _find(f"/kaggle/working/feats_{TAG}", f"feats_{TAG}")
OUT = f"/kaggle/working/cls14_out_{TAG}"
K = 32
FOLDS = [1] if QUICK else [1, 2, 3, 4]
SEEDS = [0] if QUICK else [0, 1, 2]
HEADS = ["linear"] if QUICK else ["linear", "mil_topk"]
EPOCHS = 5 if QUICK else 80
BATCH, LR, WD, TOPK = 32, 1e-3, 1e-2, 3
VAL_FRAC = 0.15


# ---------------- metrics ----------------
def confusion(y, p, n):
    cm = np.zeros((n, n), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm


def macro_f1(y, p, n):
    cm = confusion(y, p, n)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(f1.mean())


def per_class(cm):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    pr, rc = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    return pr, rc, 2 * pr * rc / np.maximum(pr + rc, 1e-9), cm.sum(1)


# ---------------- data ----------------
def read_list(p):
    out = []
    for l in open(p, errors="ignore"):
        t = l.strip().split()
        if t:
            out.append((os.path.splitext(os.path.basename(t[0]))[0], t[0].split("/")[0]))
    return out


# ---------------- model ----------------
def build(head, nc, d=2048):
    import torch, torch.nn as nn
    class H(nn.Module):
        def __init__(s):
            super().__init__()
            s.drop, s.fc = nn.Dropout(0.5), nn.Linear(d, nc)
        def forward(s, x):                      # x: B,K,d
            x = s.drop(x)
            if head == "linear":
                return s.fc(x.mean(1))
            return s.fc(x).topk(min(TOPK, x.shape[1]), dim=1).values.mean(1)
    return H()


def run(head, seed, Xtr, ytr, Xva, yva, Xte, nc, dev, wrs=None):
    wrs = USE_WRS if wrs is None else wrs
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    m = build(head, nc).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    Xg, yg = Xtr.to(dev), torch.from_numpy(ytr).to(dev)
    w = 1.0 / np.bincount(ytr, minlength=nc)[ytr]; w /= w.sum()
    best, best_state = -1, None
    for ep in range(EPOCHS):
        m.train()
        order = rng.choice(len(ytr), len(ytr), p=w) if wrs else rng.permutation(len(ytr))
        for b in range(0, len(order), BATCH):
            i = order[b:b + BATCH]
            x = Xg[i]
            sub = np.sort(rng.choice(K, K // 2, replace=False))          # random half of segments
            loss = F.cross_entropy(m(x[:, sub]), yg[i], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            f1 = macro_f1(yva, m(Xva.to(dev)).argmax(1).cpu().numpy(), nc)
        if f1 > best:
            best, best_state = f1, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        lg = m(Xte.to(dev)).cpu()
    return lg, best


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("MODE:", "QUICK (sirf test, paper ke liye nahi!)" if QUICK else "FULL")
    os.makedirs(OUT, exist_ok=True)
    ar = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Action_Regnition_splits")
    classes = [l.split()[0] for l in open(os.path.join(ar, "ClassIDs.txt")) if l.strip()]
    cid = {c: i for i, c in enumerate(classes)}; nc = len(classes)
    print(f"{nc} classes | device {dev} | USE_WRS {USE_WRS}")
    cache = {}
    def feat(n):
        if n not in cache:
            cache[n] = np.load(os.path.join(FEAT, n + ".npy")).astype(np.float32)
        return cache[n]

    res = {h: [] for h in HEADS}; pooled = {h: {"y": [], "p": []} for h in HEADS}
    pack, meta = {}, dict(classes=classes, folds={})
    for fold in FOLDS:
        tr_all = read_list(os.path.join(ar, f"train_00{fold}.txt")); te = read_list(os.path.join(ar, f"test_00{fold}.txt"))
        assert not ({n for n, _ in tr_all} & {n for n, _ in te}), "train/test overlap!"
        rs = np.random.RandomState(100 + fold); tr, va = [], []
        for c in classes:
            idx = [i for i, (_, k) in enumerate(tr_all) if k == c]; rs.shuffle(idx)
            nv = max(1, int(round(len(idx) * VAL_FRAC))); va += idx[:nv]; tr += idx[nv:]
        stack = lambda names: torch.from_numpy(np.stack([feat(n) for n in names]))
        Xtr, Xva, Xte = stack([tr_all[i][0] for i in tr]), stack([tr_all[i][0] for i in va]), stack([n for n, _ in te])
        ytr, yva = np.array([cid[tr_all[i][1]] for i in tr]), np.array([cid[tr_all[i][1]] for i in va])
        yte = np.array([cid[k] for _, k in te])
        mu, sd = Xtr.reshape(-1, 2048).mean(0), Xtr.reshape(-1, 2048).std(0) + 1e-6
        Xtr, Xva, Xte = [(x - mu) / sd for x in (Xtr, Xva, Xte)]
        print(f"\nfold {fold}: train {len(tr)} | val {len(va)} | test {len(te)}")
        pack[f"fold{fold}_y"] = yte; meta["folds"][str(fold)] = [n for n, _ in te]
        for head in HEADS:
            for seed in SEEDS:
                t0 = time.time()
                lg, bv = run(head, seed, Xtr, ytr, Xva, yva, Xte, nc, dev)
                pred = lg.argmax(1).numpy()
                top3 = float(np.mean([yte[i] in np.argsort(-lg[i].numpy())[:3] for i in range(len(yte))]))
                r = dict(fold=fold, seed=seed, val_f1=bv, acc=float((pred == yte).mean()), macro_f1=macro_f1(yte, pred, nc), top3=top3)
                res[head].append(r); pack[f"fold{fold}_{head}_s{seed}"] = lg.numpy()
                if seed == SEEDS[0]:
                    pooled[head]["y"] += yte.tolist(); pooled[head]["p"] += pred.tolist()
                print(f"  [{head} f{fold} s{seed}] {time.time() - t0:.0f}s | val F1 {bv:.3f} | acc {r['acc']:.3f} | macro-F1 {r['macro_f1']:.3f} | top-3 {top3:.3f}")

    print("\nSUMMARY (mean +- std over folds; har fold ka seed-average)")
    for head, rows in res.items():
        per_fold = {k: [np.mean([r[k] for r in rows if r["fold"] == f]) for f in FOLDS] for k in ("acc", "macro_f1", "top3")}
        print(f"{head:9s} acc {100 * np.mean(per_fold['acc']):.1f} +- {100 * np.std(per_fold['acc']):.1f} | "
              f"macro-F1 {100 * np.mean(per_fold['macro_f1']):.1f} +- {100 * np.std(per_fold['macro_f1']):.1f} | "
              f"top-3 {100 * np.mean(per_fold['top3']):.1f} +- {100 * np.std(per_fold['top3']):.1f}   (chance 7.1)")
    for head in HEADS:
        cm = confusion(np.array(pooled[head]["y"]), np.array(pooled[head]["p"]), nc)
        pr, rc, f1, sup = per_class(cm)
        print(f"\nPER-CLASS ({head}, seed {SEEDS[0]}, pooled folds)")
        print(f"{'class':22s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>5s}")
        for i, c in enumerate(classes):
            print(f"{c:22s} {pr[i]:6.2f} {rc[i]:6.2f} {f1[i]:6.2f} {sup[i]:5d}")
        print("CONFUSION (rows = true, cols = pred), class order:", classes)
        print(cm)
        np.savetxt(os.path.join(OUT, f"confusion_{head}.csv"), cm, fmt="%d", delimiter=",")

    # ---------------- DET: cascade protocol (13 anomaly classes) ----------------
    sp = os.path.join(ROOT, "UCF_Crimes-Train-Test-Split", "Anomaly_Detection_splits")
    rd = lambda p: [(os.path.splitext(os.path.basename(t[0]))[0], t[0].split("/")[0]) for t in (l.split() for l in open(p, errors="ignore")) if t and "Normal" not in t[0].split("/")[0]]
    dtr_all, dte = rd(os.path.join(ROOT, "Anomaly_Train.txt")), rd(os.path.join(sp, "Anomaly_Test.txt"))
    dcls = sorted({k for _, k in dtr_all}); dcid = {c: i for i, c in enumerate(dcls)}; dn = len(dcls)
    assert dn == 13 and {k for _, k in dte} <= set(dcls), f"anomaly classes {dn}: {dcls}"
    assert not ({n for n, _ in dtr_all} & {n for n, _ in dte}), "train/test overlap!"
    rs = np.random.RandomState(7); dtr, dva = [], []
    for c in dcls:
        idx = [i for i, (_, k) in enumerate(dtr_all) if k == c]; rs.shuffle(idx)
        nv = max(1, int(round(len(idx) * VAL_FRAC))); dva += idx[:nv]; dtr += idx[nv:]
    stack = lambda names: torch.from_numpy(np.stack([feat(n) for n in names]))
    Xtr, Xva, Xte = stack([dtr_all[i][0] for i in dtr]), stack([dtr_all[i][0] for i in dva]), stack([n for n, _ in dte])
    ytr, yva = np.array([dcid[dtr_all[i][1]] for i in dtr]), np.array([dcid[dtr_all[i][1]] for i in dva])
    yte = np.array([dcid[k] for _, k in dte])
    D = Xtr.shape[-1]
    mu, sd = Xtr.reshape(-1, D).mean(0), Xtr.reshape(-1, D).std(0) + 1e-6
    np.savez(os.path.join(OUT, "norm_det.npz"), mu=mu.numpy(), sd=sd.numpy())
    Xtr, Xva, Xte = [(x - mu) / sd for x in (Xtr, Xva, Xte)]
    print(f"\nDET: train {len(dtr)} | val {len(dva)} | test {len(dte)} anomaly videos, {dn} classes (test min/max per class {np.bincount(yte).min()}/{np.bincount(yte).max()}, train {np.bincount(ytr).min()}/{np.bincount(ytr).max()})")
    pack["det_y"] = yte; meta["det_classes"] = dcls; meta["det_names"] = [n for n, _ in dte]
    det_res, det_pooled = {}, {}
    for head in (["linear"] if QUICK else ["linear", "mil_topk"]):
        for wrs in ([False] if QUICK else [False, True]):
            key = f"{head}{'_wrs' if wrs else ''}"; rows = []
            for seed in SEEDS:
                lg, bv = run(head, seed, Xtr, ytr, Xva, yva, Xte, dn, dev, wrs=wrs)
                pred = lg.argmax(1).numpy()
                top3 = float(np.mean([yte[i] in np.argsort(-lg[i].numpy())[:3] for i in range(len(yte))]))
                rows.append(dict(seed=seed, val_f1=bv, acc=float((pred == yte).mean()), macro_f1=macro_f1(yte, pred, dn), top3=top3))
                pack[f"det_{key}_s{seed}"] = lg.numpy()
                if seed == SEEDS[0]:
                    det_pooled[key] = (yte.copy(), pred.copy())
            det_res[key] = rows
            ms = lambda k: f"{100 * np.mean([r[k] for r in rows]):.1f} +- {100 * np.std([r[k] for r in rows]):.1f}"
            print(f"  DET {key:14s} acc {ms('acc')} | macro-F1 {ms('macro_f1')} | top-3 {ms('top3')}   (chance {100 / dn:.1f})")
    # ---------------- 4 crime groups: alag head sirf 4 groups par train (sir ka kaam) ----------------
    GROUPS = {"Violence": ["Abuse", "Assault", "Fighting", "Shooting", "Arrest"], "Theft": ["Burglary", "Robbery", "Shoplifting", "Stealing"],
              "Fire/damage": ["Arson", "Explosion", "Vandalism"], "Road accident": ["RoadAccidents"]}
    gid = {c: i for i, g in enumerate(GROUPS) for c in GROUPS[g]}
    gtr, gva, gte = [np.array([gid[dcls[k]] for k in y]) for y in (ytr, yva, yte)]
    pack["grp_y"] = gte; meta["groups"] = GROUPS; grp_res = {}
    print(f"\nGROUPS (4, trained directly): test per group {np.bincount(gte, minlength=4).tolist()} | largest-group chance {100 * np.bincount(gte).max() / len(gte):.1f}")
    for head in (["linear"] if QUICK else ["linear", "mil_topk"]):
        for wrs in ([False] if QUICK else [False, True]):
            key = f"{head}{'_wrs' if wrs else ''}"; rows = []
            for seed in SEEDS:
                lg, bv = run(head, seed, Xtr, gtr, Xva, gva, Xte, 4, dev, wrs=wrs)
                pred = lg.argmax(1).numpy()
                rows.append(dict(seed=seed, val_f1=bv, acc=float((pred == gte).mean()), macro_f1=macro_f1(gte, pred, 4)))
                pack[f"grp_{key}_s{seed}"] = lg.numpy()
                if seed == SEEDS[0]:
                    np.savetxt(os.path.join(OUT, f"confusion_grp_{key}.csv"), confusion(gte, pred, 4), fmt="%d", delimiter=",")
            grp_res[key] = rows
            ms = lambda k: f"{100 * np.mean([r[k] for r in rows]):.1f} +- {100 * np.std([r[k] for r in rows]):.1f}"
            print(f"  GROUPS {key:14s} acc {ms('acc')} | macro-F1 {ms('macro_f1')} | val macro-F1 {ms('val_f1')}")
    res["_groups"] = grp_res
    for key, (yy, pp) in det_pooled.items():
        cm = confusion(yy, pp, dn)
        np.savetxt(os.path.join(OUT, f"confusion_det_{key}.csv"), cm, fmt="%d", delimiter=",")
    res["_det"] = det_res
    np.savez_compressed(os.path.join(OUT, "cls14_pack.npz"), **pack)
    json.dump(meta, open(os.path.join(OUT, "cls14_meta.json"), "w"))
    json.dump(res, open(os.path.join(OUT, "cls14_results.json"), "w"), indent=1)
    print("\nQUICK TEST OK" if QUICK else "\nDONE", "| END OF REPORT")


if __name__ == "__main__":
    main()
