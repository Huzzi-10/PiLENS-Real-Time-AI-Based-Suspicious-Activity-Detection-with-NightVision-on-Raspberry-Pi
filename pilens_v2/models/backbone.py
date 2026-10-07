# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""X3D-S (Kinetics-400, pytorchvideo) as a 2048-d clip feature extractor.

Input (B, 3, 13, 160, 160) normalised clips -> output (B, 2048) pre-logit
features. The 400-way Kinetics projection and softmax are removed.
"""

import torch.nn as nn

X3D_S_URL = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth"


def load_x3d_s(pretrained=True, weights=None):
    """Return X3D-S with the classifier removed.

    weights: optional local path to X3D_S.pyth (for offline Kaggle / Pi builds).
    pretrained=False gives random weights (only for tests / export dry-runs).
    """
    import torch
    from pytorchvideo.models.x3d import create_x3d

    model = create_x3d(input_clip_length=13, input_crop_size=160, depth_factor=2.2,
                       model_num_class=400)
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state", state))
    elif pretrained:
        state = torch.hub.load_state_dict_from_url(X3D_S_URL, map_location="cpu", progress=True)
        model.load_state_dict(state["model_state"])
    head = model.blocks[-1]
    head.dropout = None
    head.proj = None
    head.activation = None
    return model.eval()


class TinyBackbone(nn.Module):
    """Small stand-in with the same input/output contract (tests, CI, dry runs)."""

    def __init__(self, out_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, stride=(1, 4, 4), padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(8, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_backbone(name="x3d_s", pretrained=True, weights=None):
    if name == "x3d_s":
        return load_x3d_s(pretrained, weights)
    if name == "tiny":
        return TinyBackbone().eval()
    raise ValueError(f"unknown backbone {name!r} (x3d_s, tiny)")
