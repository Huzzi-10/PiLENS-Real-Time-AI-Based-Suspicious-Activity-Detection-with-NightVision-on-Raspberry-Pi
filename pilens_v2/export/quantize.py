# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Step 9: INT8 static quantization of the backbone (only after the FP32 number).

Calibration clips must come from BOTH day and night videos:
    python -m pilens_v2.export.quantize --model exports/backbone.onnx \
        --calib-list calib_videos.txt --out exports/backbone_int8.onnx

calib_videos.txt: one video path per line (e.g. 20 day + 20 night).
Report the accuracy drop by re-running the evaluation with the INT8 model's
features, and the latency with bench/benchmark.py --model backbone_int8.onnx.
"""

import argparse
from pathlib import Path

import numpy as np

from .. import spec
from ..preprocess import clip_indices, frames_to_clip


def calibration_clips(videos, clips_per_video=4, gray3=False):
    import cv2
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for c in np.linspace(0.1 * n, 0.9 * n, clips_per_video):
            frames = []
            for i in clip_indices(c, n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if ok:
                    frames.append(f)
            if len(frames) == spec.CLIP_LEN:
                yield frames_to_clip(frames, gray3)[None]
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="exports/backbone.onnx")
    ap.add_argument("--out", default="exports/backbone_int8.onnx")
    ap.add_argument("--calib-list", required=True)
    ap.add_argument("--clips-per-video", type=int, default=4)
    ap.add_argument("--gray3", action="store_true")
    ap.add_argument("--per-channel", action="store_true", default=True)
    args = ap.parse_args()

    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    videos = [line.strip() for line in Path(args.calib_list).read_text().splitlines() if line.strip()]
    clips = list(calibration_clips(videos, args.clips_per_video, args.gray3))
    if not clips:
        raise SystemExit("no calibration clips could be read")
    print(f"{len(clips)} calibration clips from {len(videos)} videos")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(clips)

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"clip": x.astype(np.float32)}

    pre = Path(args.out).with_suffix(".pre.onnx")
    quant_pre_process(args.model, str(pre))
    quantize_static(str(pre), args.out, Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=args.per_channel)
    pre.unlink(missing_ok=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
