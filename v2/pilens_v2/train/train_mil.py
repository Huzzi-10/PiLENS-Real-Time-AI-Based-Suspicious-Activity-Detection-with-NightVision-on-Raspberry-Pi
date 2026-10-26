# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Step 3 / Task B (B3): weakly supervised binary MIL head (stage 1).

Official anomaly split (1610 train / 290 test), 15% of train held out at
video level for validation. Checkpoint and threshold are chosen on validation
only; the test set is scored once with those fixed settings.

    python -m pilens_v2.train.train_mil --feats /kaggle/input/feats-x3ds --splits splits \
        --test-ann /kaggle/input/ufc-crime-full-dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
        --head mlp --seeds 0 1 2 3 4 --out runs/mil_mlp

Primary numbers are unsmoothed; the Gaussian-smoothed (sigma = 1 segment)
frame AUC is reported as a separate row.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from .. import spec
from ..data.annotations import frame_labels, read_events_csv, read_official_temporal
from ..eval.metrics import binary_report, mean_std, precision_recall_at, select_threshold
from .common import FeatureStore, read_ids, set_seed, write_json


def scores_for(model, store, ids, device, batch=64):
    import torch
    model.eval()
    out = {}
    with torch.no_grad():
        for s in range(0, len(ids), batch):
            chunk = ids[s:s + batch]
            sc = model(torch.from_numpy(store.stack(chunk)).to(device)).cpu().numpy()
            out.update(zip(chunk, sc))
    return out


def train_one(store, train_ids, val_ids, head_name, seed, args, device, val_gt=None):
    import torch
    from torch.utils.data import RandomSampler

    from ..models.heads import BINARY_HEADS, mil_ranking_loss

    set_seed(seed)
    anom = [v for v in train_ids if store.label(v) != spec.NORMAL_CLASS]
    norm = [v for v in train_ids if store.label(v) == spec.NORMAL_CLASS]
    xa, xn = torch.from_numpy(store.stack(anom)), torch.from_numpy(store.stack(norm))
    g = torch.Generator().manual_seed(seed)
    # equal anomaly / normal bags per step (balanced sampling at loader level)
    sa = iter(RandomSampler(range(len(anom)), replacement=True, num_samples=args.iters * args.batch, generator=g))
    sn = iter(RandomSampler(range(len(norm)), replacement=True, num_samples=args.iters * args.batch, generator=g))

    model = BINARY_HEADS[head_name](dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    val_labels = {v: int(store.label(v) != spec.NORMAL_CLASS) for v in val_ids}

    best = (-1.0, 0, None)
    for it in range(1, args.iters + 1):
        model.train()
        ia = torch.tensor([next(sa) for _ in range(args.batch)])
        inn = torch.tensor([next(sn) for _ in range(args.batch)])
        loss = mil_ranking_loss(model(xa[ia].to(device)), model(xn[inn].to(device)),
                                k=args.topk, lambda_smooth=args.l_smooth, lambda_sparse=args.l_sparse)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % args.eval_every == 0 or it == args.iters:
            sc = scores_for(model, store, val_ids, device)
            if val_gt:
                rep = binary_report(sc, val_labels, val_gt, {v: store.num_frames(v) for v in val_ids})
                key = rep["frame_auc"]
            else:
                key = binary_report(sc, val_labels)["video_auc"]
            if key > best[0]:
                best = (key, it, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])

    # threshold on validation, per clip score (the Pi alerts on single clips)
    sc = scores_for(model, store, val_ids, device)
    thr, f1 = select_threshold([val_labels[v] for v in val_ids], [float(np.max(sc[v])) for v in val_ids])
    return model, {"best_iter": best[1], "val_select_metric": best[0], "threshold": thr, "val_f1_at_thr": f1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--test-ann", required=True, help="Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--val-ann", default=None, help="manual annotation CSV: select on val frame AUC")
    ap.add_argument("--head", default="mlp", choices=["mlp", "tconv"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32, help="anomaly bags (and normal bags) per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--l-smooth", type=float, default=8e-5)
    ap.add_argument("--l-sparse", type=float, default=8e-5)
    ap.add_argument("--smooth-sigma", type=float, default=1.0, help="fixed in advance, separate row")
    ap.add_argument("--l2norm", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/mil")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(args.feats, args.meta, args.l2norm)
    ids = {p: read_ids(f"{args.splits}/anomaly_{p}.txt") for p in ("train", "val", "test")}
    store.check(sum(ids.values(), []))

    events = read_official_temporal(args.test_ann)
    nf = {v: store.num_frames(v) for v in ids["test"]}
    test_gt = {v: frame_labels(events.get(v, []), nf[v]) for v in ids["test"]}
    test_labels = {v: int(store.label(v) != spec.NORMAL_CLASS) for v in ids["test"]}
    no_gt = [v for v in ids["test"] if test_labels[v] and not events.get(v)]
    if no_gt:
        print(f"WARNING: {len(no_gt)} anomaly test videos have no temporal annotation, e.g. {no_gt[:3]}")

    val_gt = None
    if args.val_ann:
        ev, _ = read_events_csv(args.val_ann)
        val_gt = {v: frame_labels(ev.get(v, []), store.num_frames(v)) for v in ids["val"]}

    runs, t0 = [], time.time()
    for seed in args.seeds:
        model, info = train_one(store, ids["train"], ids["val"], args.head, seed, args, device, val_gt)
        sc = scores_for(model, store, ids["test"], device)
        rep = binary_report(sc, test_labels, test_gt, nf)
        rep_s = binary_report(sc, test_labels, test_gt, nf, smooth_sigma=args.smooth_sigma)
        pr = precision_recall_at([test_labels[v] for v in ids["test"]],
                                 [float(np.max(sc[v])) for v in ids["test"]], info["threshold"])
        runs.append({"seed": seed, **info, **rep, "frame_auc_smoothed": rep_s["frame_auc"],
                     "frame_ap_smoothed": rep_s["frame_ap"], "test_video_pr_at_val_thr": pr})
        torch.save({"head": args.head, "kind": "binary", "dropout": args.dropout, "l2norm": args.l2norm,
                    "state_dict": model.state_dict(), "threshold": info["threshold"]},
                   out / f"seed{seed}.pt")
        print(f"seed {seed}: iter {info['best_iter']:5d}  video AUC {rep['video_auc']:.3f}  "
              f"frame AUC {rep['frame_auc']:.3f} (smoothed {rep_s['frame_auc']:.3f})  AP {rep['frame_ap']:.3f}  "
              f"thr {info['threshold']:.3f}")

    keys = ["video_auc", "frame_auc", "frame_ap", "frame_auc_smoothed", "frame_ap_smoothed"]
    summary = {"head": args.head, "seeds": args.seeds, "args": vars(args),
               **{k: mean_std([r[k] for r in runs]) for k in keys},
               "runs": runs, "minutes": (time.time() - t0) / 60}
    write_json(out / "summary.json", summary)
    print("\n" + "  ".join(f"{k} {summary[k][0]:.3f} +- {summary[k][1]:.3f}" for k in keys))


if __name__ == "__main__":
    main()
