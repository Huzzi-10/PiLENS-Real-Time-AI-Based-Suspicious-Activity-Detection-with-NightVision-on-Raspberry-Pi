"""PiLENS paper_eval.py: builds all paper tables and curves from the saved Kaggle outputs (eval_pack.npz etc.). Needs only numpy (CPU, runs anywhere).
Input folders (outputs of step 2 / step 3):
  --x3  binary_out            (X3D-S)        --mv  binary_out_movinet_a0_v2   (MoViNet-A0)
  --x3c cls14_out             (X3D stage 2)  --mvc cls14_out_movinet_a0_v2
  --cnn binary_out_mnv3l      (2D CNN + temporal baselines)
Every input is optional; each folder given is reported. Output: --out paper_out/ (paper_tables.md, roc_*.csv, pr_*.csv, *.png)
Protocol: threshold from validation only (Youden); test used once. Primary run = mlp seed 0 (the deployed head); mean +- std over 5 seeds separately.
Video score = max over 32 segment scores. CI = bootstrap over test videos (B=2000, fixed seed).
Reproduce results/paper_final/paper_tables.md (from the v3 folder):
  python evaluation/paper_eval.py --x3 results/x3ds/binary_out --mv results/movinet_a0_v2/binary_out_movinet_a0_v2 \
      --x3c results/x3ds/cls14_out --mvc results/movinet_a0_v2/cls14_out_movinet_a0_v2 --cnn results/mnv3l/binary_out_mnv3l --out paper_out
"""
import os, re, json, argparse
import numpy as np


# ---------- metrics ----------
def auc_score(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); r = (((cum - cnt + 1) + cum) / 2.0)[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ap_score(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort"); s, y = s[o], y[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    ends = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp, fp = tp[ends], fp[ends]
    return float(np.sum(np.diff(np.r_[0.0, rec := tp / y.sum()]) * (tp / (tp + fp))))


def curves(y, s):
    """ROC (fpr,tpr) aur PR (recall,precision) points."""
    o = np.argsort(-s, kind="mergesort"); y = np.asarray(y)[o]; s = s[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    ends = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp, fp = tp[ends], fp[ends]
    return (np.r_[0, fp / max(fp[-1], 1)], np.r_[0, tp / max(tp[-1], 1)]), (np.r_[0, tp / max(tp[-1], 1)], np.r_[1, tp / np.maximum(tp + fp, 1)])


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def youden(vmax, y):
    cand = np.unique(vmax)
    return float(max(cand, key=lambda t: ((vmax >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) - ((vmax >= t) & (y == 0)).sum() / max((y == 0).sum(), 1)))


def gsmooth(x, sigma=1.0):
    r = int(4 * sigma + 0.5); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    return np.convolve(np.pad(x, (r, r), mode="reflect"), k, mode="valid")


def frame_metrics(ts, frames, ann, names, K=32):
    gt, sc = [], []
    for i, n in enumerate(names):
        nf = int(frames[i]); g = np.zeros(nf, np.uint8)
        for a, b in ann.get(n, []):
            g[max(a, 0):min(b + 1, nf)] = 1
        gt.append(g); sc.append(ts[i][np.minimum((np.arange(nf) * K) // nf, K - 1)])
    gt, sc = np.concatenate(gt), np.concatenate(sc)
    return auc_score(gt, sc), ap_score(gt, sc)


def rule_alert(seg, thr, k, n):
    """n lagataar segments mein se kam az kam k thr se upar."""
    hit = (seg >= thr).astype(int)
    if n == 1:
        return bool(hit.max() >= k)
    w = np.convolve(hit, np.ones(n, int), mode="valid")
    return bool(w.max() >= k) if len(w) else False


def category(name):
    m = re.match(r"[A-Za-z]+", name)
    return m.group(0) if m else name


# ---------- loading ----------
class BB:
    def __init__(s, d, label):
        s.label = label
        s.pack = dict(np.load(os.path.join(d, "eval_pack.npz")))
        s.meta = json.load(open(os.path.join(d, "test_meta.json")))
        s.names, s.y = s.meta["names"], np.array(s.meta["y"])
        s.frames, s.ann = s.pack["test_frames"], {k: [tuple(x) for x in v] for k, v in s.meta["ann"].items()}
        s.vy = s.pack["val_y"]
        s.abl = json.load(open(os.path.join(d, "ablation_results.json"))) if os.path.exists(os.path.join(d, "ablation_results.json")) else None

    def run(s, head, seed):
        return s.pack[f"val_{head}_{seed}"], s.pack[f"test_{head}_{seed}"], float(s.pack[f"thr_{head}_{seed}"])


def row_metrics(bb, vs, ts, thr):
    p, r, f1 = prf(bb.y, (ts.max(1) >= thr).astype(int))
    fa, fp_ = frame_metrics(ts, bb.frames, bb.ann, bb.names)
    return dict(video_auc=auc_score(bb.y, ts.max(1)), frame_auc=fa, frame_ap=fp_, prec=p, rec=r, f1=f1)


def ms(rows, k, pct=False):
    v = np.array([r[k] for r in rows]) * (100 if pct else 1)
    return f"{v.mean():.2f} ± {v.std():.2f}" if pct else f"{v.mean():.3f} ± {v.std():.3f}"


def boot_auc(y, s_list, B=2000, seed=0):
    """Har backbone ke video scores par SAME resampling (paired). Return: CI har ek ka, aur pehle-doosre ka farq."""
    rs = np.random.RandomState(seed); n = len(y); out = np.zeros((len(s_list), B))
    for b in range(B):
        i = rs.randint(0, n, n)
        if y[i].min() == y[i].max():
            out[:, b] = np.nan; continue
        for j, s in enumerate(s_list):
            out[j, b] = auc_score(y[i], s[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x3"); ap.add_argument("--mv"); ap.add_argument("--x3c"); ap.add_argument("--mvc")
    ap.add_argument("--cnn", help="binary_out_mnv3l (c_step2: MobileNetV3 + lstm/transformer/meanpool)")
    ap.add_argument("--out", default="paper_out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    L = []
    P = lambda *x: (print(*x), L.append(" ".join(str(v) for v in x)))
    bbs = [BB(d, lab) for d, lab in ((a.x3, "X3D-S"), (a.mv, "MoViNet-A0")) if d]
    for b in bbs:
        P(f"\n## {b.label}: test {len(b.y)} videos (anomaly {int(b.y.sum())}), val {len(b.vy)}")
    heads = sorted({k.split("_")[1] for b in bbs for k in b.pack if k.startswith("test_") and not k.startswith("test_abl")})

    # ---- 1. main table ----
    P("\n### Table 1. Binary detection (mean ± std over seeds; thr = val Youden)")
    P("| backbone | head | video AUC | frame AUC | frame AP | precision | recall | F1 | n seeds |\n|---|---|---|---|---|---|---|---|---|")
    allrows = {}
    for b in bbs:
        for h in [x for x in ("mlp", "tconv") if x in heads]:
            seeds = sorted({int(k.split("_")[2]) for k in b.pack if k.startswith(f"test_{h}_")})
            if not seeds:
                continue
            rows = [row_metrics(b, *b.run(h, sd)) for sd in seeds]; allrows[(b.label, h)] = rows
            P(f"| {b.label} | {h} | {ms(rows, 'video_auc')} | {ms(rows, 'frame_auc')} | {ms(rows, 'frame_ap')} | {ms(rows, 'prec')} | {ms(rows, 'rec')} | {ms(rows, 'f1')} | {len(seeds)} |")

    # ---- 2. bootstrap CI + paired diff (mlp seed 0) ----
    if all(("mlp" in heads) for _ in bbs) and bbs:
        S = [b.run("mlp", 0)[1].max(1) for b in bbs]
        if len(bbs) == 2:
            assert bbs[0].names == bbs[1].names, "test videos ka order alag hai"
        bo = boot_auc(bbs[0].y, S)
        P("\n### Table 2. Video AUC, mlp seed 0, bootstrap 95% CI (test videos par, B=2000)")
        for j, b in enumerate(bbs):
            P(f"- {b.label}: {auc_score(b.y, S[j]):.3f}  [{np.nanpercentile(bo[j], 2.5):.3f}, {np.nanpercentile(bo[j], 97.5):.3f}]")
        if len(bbs) == 2:
            d = bo[1] - bo[0]
            P(f"- paired difference ({bbs[1].label} - {bbs[0].label}): {np.nanmean(d):+.3f}  [{np.nanpercentile(d, 2.5):+.3f}, {np.nanpercentile(d, 97.5):+.3f}]  ({'significant' if np.nanpercentile(d, 2.5) > 0 or np.nanpercentile(d, 97.5) < 0 else 'NOT significant'} at 95%)")

        # ---- curves ----
        for j, b in enumerate(bbs):
            (fpr, tpr), (rec, prc) = curves(b.y, S[j])
            tag = b.label.replace(" ", "").replace("-", "")
            np.savetxt(os.path.join(a.out, f"roc_{tag}.csv"), np.c_[fpr, tpr], delimiter=",", header="fpr,tpr", comments="")
            np.savetxt(os.path.join(a.out, f"pr_{tag}.csv"), np.c_[rec, prc], delimiter=",", header="recall,precision", comments="")

        # ---- 3. operating points ----
        P("\n### Table 3. Operating points (mlp seed 0; thr val se, test par sirf lagaya)")
        P("| backbone | thr rule | thr | video TPR | video FPR (normal test) | clip FPR (normal segments) |\n|---|---|---|---|---|---|")
        for b in bbs:
            vs, ts, thr0 = b.run("mlp", 0); vm, tm = vs.max(1), ts.max(1)
            ops = [("Youden (deployed)", thr0)]
            for fpr_t in (0.05, 0.10):
                ops.append((f"val FPR={int(fpr_t * 100)}%", float(np.quantile(vm[b.vy == 0], 1 - fpr_t))))
            for nm, t in ops:
                tpr_ = float((tm[b.y == 1] >= t).mean()); fpr_ = float((tm[b.y == 0] >= t).mean()); cfpr = float((ts[b.y == 0] >= t).mean())
                P(f"| {b.label} | {nm} | {t:.3f} | {tpr_:.3f} | {fpr_:.3f} | {cfpr:.4f} |")

        # ---- 4. alert rule ----
        P("\n### Table 4. Alert rule (n lagataar segments mein se k, thr = Youden, test videos par)")
        P("| backbone | rule | video TPR | video FPR |\n|---|---|---|---|")
        for b in bbs:
            vs, ts, thr0 = b.run("mlp", 0)
            for k, n in ((1, 1), (2, 2), (2, 3), (3, 5)):
                al = np.array([rule_alert(ts[i], thr0, k, n) for i in range(len(b.y))])
                P(f"| {b.label} | {k} of {n} | {al[b.y == 1].mean():.3f} | {al[b.y == 0].mean():.3f} |")
        P("(note: UCF test mein segments poori video par phaile hain, Pi ke 0.6 s hop se alag; ye rule ka relative asar dikhata hai, per-hour rate nahi.)")

        # ---- 5. per-category ----
        P("\n### Table 5. Per-category detection (mlp seed 0, Youden thr): recall aur AUC vs normal")
        for b in bbs:
            vs, ts, thr0 = b.run("mlp", 0); tm = ts.max(1)
            cats = np.array([category(n) for n in b.names]); nm = b.y == 0
            P(f"\n{b.label}:\n| category | n | recall | AUC vs normal |\n|---|---|---|---|")
            for c in sorted(set(cats[b.y == 1])):
                m = (cats == c) & (b.y == 1)
                yy = np.r_[np.ones(m.sum(), int), np.zeros(nm.sum(), int)]
                P(f"| {c} | {int(m.sum())} | {float((tm[m] >= thr0).mean()):.2f} | {auc_score(yy, np.r_[tm[m], tm[nm]]):.3f} |")

        # ---- 6. ensemble ----
        if len(bbs) == 2 and np.array_equal(bbs[0].vy, bbs[1].vy):
            P("\n### Table 6. Score-level ensemble (X3D + MoViNet, mlp seed 0; offline, 2 backbone = ~2x latency)")
            v = (bbs[0].run("mlp", 0)[0] + bbs[1].run("mlp", 0)[0]) / 2; t = (bbs[0].run("mlp", 0)[1] + bbs[1].run("mlp", 0)[1]) / 2
            thr = youden(v.max(1), bbs[0].vy); r = row_metrics(bbs[0], v, t, thr)
            P("| " + " | ".join(f"{k} {x:.3f}" for k, x in r.items()) + " |")

    # ---- 7. ablations ----
    for b in bbs:
        if b.abl:
            P(f"\n### Table 7. Ablation, {b.label} (mlp-based, seeds 0-2)\n| variant | video AUC | frame AUC | frame AP | F1 |\n|---|---|---|---|---|")
            for nm, rows in b.abl.items():
                P(f"| {nm} | {ms(rows, 'video_auc')} | {ms(rows, 'frame_auc')} | {ms(rows, 'frame_ap')} | {ms(rows, 'f1')} |")

    # ---- 9. 2D CNN + temporal baselines (c_step2) vs 3D backbones ----
    if a.cnn:
        c = BB(a.cnn, "MobileNetV3-L")
        P(f"\n### Table 9. 2D CNN (MobileNetV3-Large, per frame) + temporal head vs 3D backbones (mean ± std over seeds; thr = val Youden)")
        P("| model | video AUC | frame AUC | frame AP | precision | recall | F1 | n seeds |\n|---|---|---|---|---|---|---|---|")
        for (lab, h), rows in allrows.items():
            if h == "mlp":
                P(f"| {lab} + MLP | {ms(rows, 'video_auc')} | {ms(rows, 'frame_auc')} | {ms(rows, 'frame_ap')} | {ms(rows, 'prec')} | {ms(rows, 'rec')} | {ms(rows, 'f1')} | {len(rows)} |")
        chs = [h for h in ("lstm", "transformer", "meanpool") if f"test_{h}_0" in c.pack]
        for h in chs:
            seeds = sorted({int(k.split("_")[2]) for k in c.pack if k.startswith(f"test_{h}_")})
            rows = [row_metrics(c, *c.run(h, sd)) for sd in seeds]
            P(f"| CNN + {h} | {ms(rows, 'video_auc')} | {ms(rows, 'frame_auc')} | {ms(rows, 'frame_ap')} | {ms(rows, 'prec')} | {ms(rows, 'rec')} | {ms(rows, 'f1')} | {len(seeds)} |")
        ref = next((b for b in bbs if b.label == "MoViNet-A0"), bbs[-1] if bbs else None)
        if ref is not None and "mlp" in heads and chs:
            assert ref.names == c.names, "test videos ka order alag hai"
            S = [ref.run("mlp", 0)[1].max(1)] + [c.run(h, 0)[1].max(1) for h in chs]
            bo = boot_auc(ref.y, S)
            P(f"Paired bootstrap, seed 0, video AUC (B=2000): {ref.label} {auc_score(ref.y, S[0]):.3f}")
            for j, h in enumerate(chs, 1):
                d = bo[j] - bo[0]
                P(f"- CNN + {h}: {auc_score(ref.y, S[j]):.3f} [{np.nanpercentile(bo[j], 2.5):.3f}, {np.nanpercentile(bo[j], 97.5):.3f}] | minus {ref.label}: {np.nanmean(d):+.3f} "
                  f"[{np.nanpercentile(d, 2.5):+.3f}, {np.nanpercentile(d, 97.5):+.3f}] ({'significant' if np.nanpercentile(d, 2.5) > 0 or np.nanpercentile(d, 97.5) < 0 else 'NOT significant'})")
            for h in chs:
                vs, ts, thr0 = c.run(h, 0); tm = ts.max(1)
                P(f"- CNN + {h} operating point (Youden): video TPR {(tm[c.y == 1] >= thr0).mean():.3f} | video FPR {(tm[c.y == 0] >= thr0).mean():.3f}")

    # ---- 8. stage 2 + cascade ----
    for (cdir, lab) in ((a.x3c, "X3D-S"), (a.mvc, "MoViNet-A0")):
        if not cdir:
            continue
        pk = dict(np.load(os.path.join(cdir, "cls14_pack.npz"))); mt = json.load(open(os.path.join(cdir, "cls14_meta.json")))
        P(f"\n### Table 8. Stage 2 ({lab})")
        folds = sorted({int(k.split("_")[0][4:]) for k in pk if k.startswith("fold") and k.endswith("_y")})
        for h in ("linear", "mil_topk"):
            accs = []
            for f in folds:
                y = pk[f"fold{f}_y"]; ks = [k for k in pk if k.startswith(f"fold{f}_{h}_s")]
                if ks:
                    accs.append(np.mean([(pk[k].argmax(1) == y).mean() for k in ks]))
            if accs:
                P(f"- official 14-class folds, {h}: acc {100 * np.mean(accs):.1f} ± {100 * np.std(accs):.1f} (chance 7.1)")
        dy = pk["det_y"]; dn = len(mt["det_classes"])
        keys = sorted({k.rsplit("_s", 1)[0] for k in pk if k.startswith("det_") and k != "det_y"})
        for key in keys:
            ks = [k for k in pk if k.startswith(key + "_s")]
            acc = [(pk[k].argmax(1) == dy).mean() for k in ks]
            t3 = [np.mean([dy[i] in np.argsort(-pk[k][i])[:3] for i in range(len(dy))]) for k in ks]
            P(f"- DET 13-class (140 anomaly test videos), {key[4:]}: acc {100 * np.mean(acc):.1f} ± {100 * np.std(acc):.1f} | top-3 {100 * np.mean(t3):.1f} (chance {100 / dn:.1f})")
        # cascade end-to-end (binary mlp seed 0 + best-named DET head)
        b = next((x for x in bbs if x.label == lab), None)
        if b is not None and keys:
            best = max(keys, key=lambda kk: np.mean([(pk[k].argmax(1) == dy).mean() for k in pk if k.startswith(kk + "_s")]))
            lg = pk[best + "_s0"]; vs, ts, thr0 = b.run("mlp", 0); tm = ts.max(1)
            idx = {n: i for i, n in enumerate(b.names)}; sel = np.array([idx[n] for n in mt["det_names"]])
            det = tm[sel] >= thr0; cor = lg.argmax(1) == dy
            P(f"\nCascade end-to-end ({lab}; binary mlp seed 0 -> {best[4:]} stage 2): anomaly videos detected {det.mean():.3f} | "
              f"class correct given detected {cor[det].mean() if det.any() else float('nan'):.3f} | detected AND correct {(det & cor).mean():.3f} | "
              f"normal test videos falsely alerted {float((tm[b.y == 0] >= thr0).mean()):.3f}")

    # ---- figure ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        for b in bbs:
            s = b.run("mlp", 0)[1].max(1); (fpr, tpr), (rec, prc) = curves(b.y, s)
            ax[0].plot(fpr, tpr, label=f"{b.label} (AUC {auc_score(b.y, s):.3f})"); ax[1].plot(rec, prc, label=b.label)
        ax[0].set(xlabel="FPR", ylabel="TPR", title="Video-level ROC"); ax[1].set(xlabel="Recall", ylabel="Precision", title="Video-level PR")
        ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(a.out, "roc_pr.png"), dpi=200)
    except Exception as e:
        P("(figure skip:", e, ")")
    open(os.path.join(a.out, "paper_tables.md"), "w").write("\n".join(L))
    print(f"\nsaved -> {a.out}/paper_tables.md (+ roc/pr csv, roc_pr.png)")


if __name__ == "__main__":
    main()
