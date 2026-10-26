# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Single source of truth for the v2 model contract (notes, sections 5 and 11).

Change a value here and every stage (feature extraction, training, ONNX export,
Pi runtime) picks it up. Never hardcode these numbers anywhere else.
"""

# --- Clip -------------------------------------------------------------------
CLIP_LEN = 13          # frames per clip (X3D-S native)
FRAME_STRIDE = 6       # frames between sampled frames at SOURCE_FPS
SOURCE_FPS = 30.0      # UCF-Crime fps; the runtime converts stride to seconds
CLIP_SPAN_SEC = (CLIP_LEN - 1) * FRAME_STRIDE / SOURCE_FPS  # 2.4 s
FRAME_STEP_SEC = FRAME_STRIDE / SOURCE_FPS                  # 0.2 s

# --- Spatial ----------------------------------------------------------------
SHORT_SIDE = 182       # resize the short side to this, then center crop
CROP_SIZE = 160
# X3D (pytorchvideo, Kinetics-400) normalisation
MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

# --- Untrimmed video sampling ----------------------------------------------
N_SEGMENTS = 32        # one clip centred in each of 32 equal temporal segments
FEATURE_DIM = 2048     # X3D-S pre-logit vector

# --- Classes ----------------------------------------------------------------
# UCF-Crime: 13 anomaly classes + Normal = 14 classes for Task A.
CLASSES_14 = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "Normal", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing",
    "Vandalism",
]
NORMAL_CLASS = "Normal"
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_14)}

# --- Alert logic defaults (tune ONLY on the validation split) ---------------
VOTE_K = 2             # "2 out of 3 clips suspicious"
VOTE_N = 3
STAGE2_CLIPS = 4       # average features of the last 3-5 clips for stage 2
STAGE2_MIN_CONF = 0.35 # below this, show "Unknown suspicious"
UNKNOWN_LABEL = "Unknown suspicious"


def spec_dict():
    """The contract as a plain dict (written next to every ONNX export)."""
    return {
        "clip_len": CLIP_LEN,
        "frame_stride": FRAME_STRIDE,
        "source_fps": SOURCE_FPS,
        "clip_span_sec": CLIP_SPAN_SEC,
        "short_side": SHORT_SIDE,
        "crop_size": CROP_SIZE,
        "mean": list(MEAN),
        "std": list(STD),
        "n_segments": N_SEGMENTS,
        "feature_dim": FEATURE_DIM,
        "classes_14": list(CLASSES_14),
    }
