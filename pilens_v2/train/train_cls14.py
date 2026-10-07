# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Step 3 / Task A (B2): 14-class head on frozen X3D-S features.

Official 4 folds x N seeds. Epoch selection on the fold's validation split
(macro-F1), then ONE test evaluation with that fixed checkpoint.

    python -m pilens_v2.train.train_cls14 --feats /kaggle/input/feats-x3ds \
        --splits splits --head linear --seeds 0 1 2 --out runs/cls14_linear

Report: mean +- std over folds (each fold averaged over seeds), per-class
precision/recall/F1 and a confusion matrix summed over folds.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from .. import spec
from ..eval.metrics import classification_report, mean_std
from .common import FeatureStore, class_balanced_sampler, read_ids, set_seed, write_json


def train_one(store, train_ids, val_ids, test_ids, head_name, seed, args, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from ..models.heads import CLS_HEADS

    set_seed(seed)
    y_of = lambda ids: np.array([spec.CLASS_TO_IDX[store.label(v)] for v in ids])
    xtr, ytr = torch.from_numpy(store.stack(train_ids)), torch.from_numpy(y_of(train_ids))
    xva, yva = torch.from_numpy(store.stack(val_ids)).to(device), y_of(val_ids)
    xte, yte = torch.from_numpy(store.stack(test_ids)).to(device), y_of(test_ids)

    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch,
                        sampler=class_balanced_sampler(ytr.numpy(), seed))
    model = CLS_HEADS[head_name](dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def predict(x):
        model.eval()
        with torch.no_grad():
            return model(x).argmax(1).cpu().numpy()

    best = (-1.0, -1.0, 0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=args.label_smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
        rep = classification_report(yva, predict(xva), len(spec.CLASSES_14))
        key = (rep["macro_f1"], rep["accuracy"])
        if key > best[:2]:
            best = (*key, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[3])
    test = classification_report(yte, predict(xte), len(spec.CLASSES_14))
    return {"best_epoch": best[2], "val_macro_f1": best[0], "val_accuracy": best[1], "test": test}, best[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--head", default="linear", choices=["linear", "mil_topk", "tconv_mil"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--l2norm", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/cls14")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    store = FeatureStore(args.feats, args.meta, args.l2norm)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs, fold_acc, fold_f1 = [], [], []
    cm_sum = np.zeros((len(spec.CLASSES_14),) * 2, dtype=np.int64)
    t0 = time.time()
    for k in args.folds:
        ids = {p: read_ids(f"{args.splits}/cls14_fold{k}_{p}.txt") for p in ("train", "val", "test")}
        store.check(sum(ids.values(), []))
        accs, f1s = [], []
        for seed in args.seeds:
            res, state = train_one(store, ids["train"], ids["val"], ids["test"], args.head, seed, args, device)
            torch.save({"head": args.head, "kind": "cls14", "dropout": args.dropout,
                        "l2norm": args.l2norm, "state_dict": state, "classes": spec.CLASSES_14},
                       out / f"fold{k}_seed{seed}.pt")
            accs.append(res["test"]["accuracy"])
            f1s.append(res["test"]["macro_f1"])
            cm_sum += np.array(res["test"]["confusion_matrix"]) if seed == args.seeds[0] else 0
            runs.append({"fold": k, "seed": seed, **res})
            print(f"fold {k} seed {seed}: epoch {res['best_epoch']:3d}  val F1 {res['val_macro_f1']:.3f}  "
                  f"test acc {res['test']['accuracy']:.3f}  F1 {res['test']['macro_f1']:.3f}")
        fold_acc.append(float(np.mean(accs)))
        fold_f1.append(float(np.mean(f1s)))

    tp = np.diag(cm_sum).astype(float)
    prec, rec = tp / np.maximum(1, cm_sum.sum(0)), tp / np.maximum(1, cm_sum.sum(1))
    summary = {
        "head": args.head, "folds": args.folds, "seeds": args.seeds, "args": vars(args),
        "accuracy_mean_std": mean_std(fold_acc), "macro_f1_mean_std": mean_std(fold_f1),
        "per_fold_accuracy": fold_acc, "per_fold_macro_f1": fold_f1,
        "per_class": {c: {"precision": float(p), "recall": float(r),
                          "f1": float(2 * p * r / (p + r)) if p + r else 0.0}
                      for c, p, r in zip(spec.CLASSES_14, prec, rec)},
        "confusion_matrix_first_seed_summed_over_folds": cm_sum.tolist(),
        "classes": spec.CLASSES_14, "runs": runs, "minutes": (time.time() - t0) / 60,
    }
    write_json(out / "summary.json", summary)
    a, f = summary["accuracy_mean_std"], summary["macro_f1_mean_std"]
    print(f"\n{args.head}: accuracy {100 * a[0]:.1f} +- {100 * a[1]:.1f}   "
          f"macro-F1 {100 * f[0]:.1f} +- {100 * f[1]:.1f}   (chance {100 / len(spec.CLASSES_14):.1f})")


if __name__ == "__main__":
    main()
