# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Step 4: export backbone + both heads to ONNX and check parity.

    python -m pilens_v2.export.export_onnx --out exports \
        --binary-ckpt runs/mil_mlp/seed0.pt --cls-ckpt runs/cls14_linear/fold1_seed0.pt \
        --parity-video /kaggle/input/ufc-crime-full-dataset/.../Fighting003_x264.mp4

Writes:
  exports/backbone.onnx      clip  (B, 3, 13, 160, 160) -> feat   (B, 2048)
  exports/binary_head.onnx   feats (B, T, 2048)         -> scores (B, T)
  exports/cls14_head.onnx    feats (B, T, 2048)         -> logits (B, 14)
  exports/pilens_v2.json     spec, threshold, classes, parity numbers, versions

The backbone runs once per clip; both heads run on cached features, so stage 2
costs almost nothing at alert time. Copy the whole exports/ folder to the Pi.
"""

import argparse
import inspect
import json
import platform
import warnings
from pathlib import Path

import numpy as np

from .. import spec


def _torch_export(model, args, path, input_names, output_names, dynamic_axes):
    import torch
    kwargs = dict(input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes, opset_version=17, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # TorchScript exporter: stable for Conv3d, opset 17
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, args, str(path), **kwargs)


def wrap_head(head, l2norm):
    """Bake the optional feature L2-norm (used in training) into the exported head."""
    import torch.nn as nn
    import torch.nn.functional as F

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head

        def forward(self, feats):
            if l2norm:
                feats = F.normalize(feats, dim=-1, eps=1e-6)
            return self.head(feats)

    return Wrapped().eval()


def load_head(ckpt, kind):
    """Head module from a training checkpoint, or an untrained one (dry run)."""
    import torch

    from ..models.heads import BINARY_HEADS, CLS_HEADS
    table = BINARY_HEADS if kind == "binary" else CLS_HEADS
    if ckpt:
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        head = table[c["head"]](dropout=c.get("dropout", 0.5))
        head.load_state_dict(c["state_dict"])
        info = {"checkpoint": str(ckpt), "head": c["head"], "l2norm": bool(c.get("l2norm", False)),
                "threshold": c.get("threshold"), "trained": True}
    else:
        name = "mlp" if kind == "binary" else "linear"
        head = table[name]()
        info = {"checkpoint": None, "head": name, "l2norm": False, "threshold": None, "trained": False}
        print(f"WARNING: no {kind} checkpoint given, exporting an UNTRAINED {name} head (dry run only)")
    return wrap_head(head.eval(), info["l2norm"]), info


def compare(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {"max_abs_diff": float(np.max(np.abs(a - b))), "cosine": cos}


def parity_clip(video, gray3=False):
    """A real clip from a video (middle of the file) for the parity check."""
    import cv2

    from ..preprocess import clip_indices, frames_to_clip
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = clip_indices(n // 2, n)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            raise IOError(f"cannot read frame {i} of {video}")
        frames.append(f)
    cap.release()
    return frames_to_clip(frames, gray3)[None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="exports")
    ap.add_argument("--backbone", default="x3d_s", choices=["x3d_s", "tiny"])
    ap.add_argument("--weights", default=None, help="local X3D_S.pyth (offline)")
    ap.add_argument("--no-pretrained", action="store_true", help="random backbone (dry run)")
    ap.add_argument("--binary-ckpt", default=None)
    ap.add_argument("--cls-ckpt", default=None)
    ap.add_argument("--parity-video", default=None)
    ap.add_argument("--gray3", action="store_true", help="record that the model expects gray->3ch input")
    ap.add_argument("--tol", type=float, default=1e-3, help="max allowed |torch - onnx| (FP32)")
    args = ap.parse_args()

    import onnxruntime as ort
    import torch

    from ..models.backbone import build_backbone

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backbone = build_backbone(args.backbone, pretrained=not args.no_pretrained, weights=args.weights)
    binary, bin_info = load_head(args.binary_ckpt, "binary")
    cls14, cls_info = load_head(args.cls_ckpt, "cls14")

    clip = torch.randn(1, 3, spec.CLIP_LEN, spec.CROP_SIZE, spec.CROP_SIZE)
    feats = torch.randn(1, spec.N_SEGMENTS, spec.FEATURE_DIM)
    _torch_export(backbone, (clip,), out / "backbone.onnx", ["clip"], ["feat"],
                  {"clip": {0: "batch"}, "feat": {0: "batch"}})
    _torch_export(binary, (feats,), out / "binary_head.onnx", ["feats"], ["scores"],
                  {"feats": {0: "batch", 1: "clips"}, "scores": {0: "batch", 1: "clips"}})
    _torch_export(cls14, (feats,), out / "cls14_head.onnx", ["feats"], ["logits"],
                  {"feats": {0: "batch", 1: "clips"}, "logits": {0: "batch"}})

    def run(name, x):
        s = ort.InferenceSession(str(out / name), providers=["CPUExecutionProvider"])
        return s.run(None, {s.get_inputs()[0].name: x})[0]

    parity, failed = {}, []
    clips = {"random": clip.numpy()}
    if args.parity_video:
        clips["video"] = parity_clip(args.parity_video, args.gray3)
    with torch.no_grad():
        for tag, x in clips.items():
            parity[f"backbone_{tag}"] = compare(backbone(torch.from_numpy(x)).numpy(), run("backbone.onnx", x))
        for t in (1, spec.VOTE_N, spec.STAGE2_CLIPS, spec.N_SEGMENTS):
            f = torch.randn(2, t, spec.FEATURE_DIM)
            parity[f"binary_T{t}"] = compare(binary(f).numpy(), run("binary_head.onnx", f.numpy()))
            if t >= 3:  # top-k heads need at least k clips
                ref, got = cls14(f).numpy(), run("cls14_head.onnx", f.numpy())
                parity[f"cls14_T{t}"] = {**compare(ref, got),
                                         "argmax_agree": bool((ref.argmax(1) == got.argmax(1)).all())}
    for name, r in parity.items():
        ok = r["max_abs_diff"] <= args.tol and r.get("argmax_agree", True)
        failed += [] if ok else [name]
        print(f"parity {name:18s} max|diff| {r['max_abs_diff']:.2e}  cos {r['cosine']:.6f}  {'OK' if ok else 'FAIL'}")

    meta = {
        "spec": spec.spec_dict(), "gray3": args.gray3, "backbone": args.backbone,
        "backbone_pretrained": bool(args.weights or not args.no_pretrained),
        "binary_head": bin_info, "cls14_head": cls_info,
        "threshold": bin_info["threshold"] if bin_info["threshold"] is not None else 0.5,
        "parity": parity, "parity_ok": not failed,
        "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__,
                     "python": platform.python_version()},
    }
    (out / "pilens_v2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}/backbone.onnx, binary_head.onnx, cls14_head.onnx, pilens_v2.json")
    if failed:
        raise SystemExit(f"PARITY FAILED: {failed}")


if __name__ == "__main__":
    main()
