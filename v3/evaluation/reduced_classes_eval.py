"""PiLENS: results with fewer classes, WITHOUT new training (from the saved logits). Runs anywhere (numpy only).
  python evaluation/reduced_classes_eval.py      -> results/paper_final/reduced_classes.md
Head: MIL top-k (the kind that runs on the Pi), the same for both models. Linear head numbers are listed alongside.
Groups were fixed in advance (semantic); the 6 "notes" classes come from the project's handwritten design notes and were chosen before seeing results.
Restriction = argmax over only these classes in the 13/14-class head logits (no separate head trained)."""
import os, json, numpy as np
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(H, "results")   # v3/results
C14 = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal_Videos_event', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
C13 = [c for c in C14 if c != 'Normal_Videos_event']
G = {'Violence': ['Abuse', 'Assault', 'Fighting', 'Shooting', 'Arrest'], 'Theft': ['Burglary', 'Robbery', 'Shoplifting', 'Stealing'],
     'Fire/damage': ['Arson', 'Explosion', 'Vandalism'], 'Road accident': ['RoadAccidents']}
NOTES = ['Abuse', 'Fighting', 'Robbery', 'Shooting', 'Shoplifting']
MODELS = [('MoViNet-A0', 'movinet_a0_v2/cls14_out_movinet_a0_v2'), ('X3D-S', 'x3ds/cls14_out')]
if os.path.exists(os.path.join(R, 'movinet_a1_v2/cls14_out_movinet_a1_v2/cls14_pack.npz')):   # A1 run aane ke baad khud shamil
    MODELS.insert(1, ('MoViNet-A1', 'movinet_a1_v2/cls14_out_movinet_a1_v2'))


def sm(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def grp(L, y, cls, extra=False):
    names = list(G) + (['Normal'] if extra else [])
    gi = {c: i for i, g in enumerate(G) for c in G[g]}
    if extra: gi['Normal_Videos_event'] = len(G)
    P = sm(L.astype(np.float64)); GP = np.zeros((len(y), len(names)))
    for j, c in enumerate(cls): GP[:, gi[c]] += P[:, j]
    yt = np.array([gi[cls[k]] for k in y]); return (GP.argmax(1) == yt).mean(), np.bincount(yt, minlength=len(names))


def sub(L, y, cls, s):
    idx = [cls.index(c) for c in s]; m = np.isin(y, idx)
    return (L[m][:, idx].argmax(1) == np.array([idx.index(k) for k in y[m]])).mean(), m.sum()


def ms(v): return f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}"


rows = {}
for name, p in MODELS:
    d = np.load(os.path.join(R, p, "cls14_pack.npz")); meta = json.load(open(os.path.join(R, p, "cls14_meta.json")))
    assert meta['classes'] == C14 and meta['det_classes'] == C13
    for h in ('mil_topk', 'linear'):
        r = {}
        L = [d[f'det_{h}_s{s}'] for s in range(3)]; y = d['det_y']
        r['det13'] = ms([(l.argmax(1) == y).mean() for l in L])
        r['det13_top3'] = ms([np.mean([y[i] in np.argsort(-l[i])[:3] for i in range(len(y))]) for l in L])
        r['det4'] = ms([grp(l, y, C13)[0] for l in L])
        r['det_notes5'] = ms([sub(l, y, C13, NOTES)[0] for l in L])
        a14, a5g, a6, a5 = [], [], [], []
        for f in range(1, 5):
            yf = d[f'fold{f}_y']
            for s in range(3):
                l = d[f'fold{f}_{h}_s{s}']
                a14.append((l.argmax(1) == yf).mean()); a5g.append(grp(l, yf, C14, True)[0])
                a6.append(sub(l, yf, C14, NOTES + ['Normal_Videos_event'])[0]); a5.append(sub(l, yf, C14, NOTES)[0])
        r['f14'], r['f4n'], r['f6'], r['f5'] = ms(a14), ms(a5g), ms(a6), ms(a5)
        rows[(name, h)] = r
    cnt_det = grp(L[0], y, C13)[1]; cnt_f = grp(d['fold1_mil_topk_s0'], d['fold1_y'], C14, True)[1]

T = [("13 crime classes, top-1 (140 anomaly test videos)", 'det13', "7.7"),
     ("13 crime classes, top-3", 'det13_top3', "23.1"),
     ("4 crime groups (140 anomaly test videos)", 'det4', f"{100 * cnt_det.max() / cnt_det.sum():.1f} (largest group)"),
     ("Official 14 classes incl. Normal (4 folds)", 'f14', "7.1"),
     ("4 crime groups + Normal (4 folds)", 'f4n', f"{100 * cnt_f.max() / cnt_f.sum():.1f} (largest group)"),
     ("Notes' 6 classes: Normal, Fighting, Robbery, Shoplifting, Shooting, Abuse (4 folds, 12 videos each)", 'f6', "16.7"),
     ("Notes' 5 crime classes, without Normal (4 folds)", 'f5', "20.0"),
     ("Notes' 5 crime classes on the 140-video test set (unbalanced: 23 Shooting, 21 Shoplifting)", 'det_notes5', "20.0")]
out = ["# Reduced-class results (no retraining, from saved logits)", "",
       "Accuracy %, mean ± std (DET: 3 seeds; folds: 4 folds x 3 seeds). Head: MIL top-k (Pi type); linear head in brackets.", "",
       "| Task | " + " | ".join(n for n, _ in MODELS) + " | Chance |", "|---" * (len(MODELS) + 2) + "|"]
for t, k, c in T:
    out.append(f"| {t} | " + " | ".join(f"{rows[(n, 'mil_topk')][k]} ({rows[(n, 'linear')][k]})" for n, _ in MODELS) + f" | {c} |")
out += ["", f"DET group sizes {dict(zip(G, cnt_det.tolist()))}; fold group sizes (Violence, Theft, Fire, Road, Normal) {cnt_f.tolist()} per fold."]
os.makedirs(os.path.join(R, "paper_final"), exist_ok=True)
open(os.path.join(R, "paper_final", "reduced_classes.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
