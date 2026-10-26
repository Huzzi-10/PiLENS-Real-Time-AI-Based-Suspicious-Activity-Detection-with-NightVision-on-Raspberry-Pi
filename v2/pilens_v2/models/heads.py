# Copyright (c) 2026 Huzaifa
# Licensed under the Apache License, Version 2.0

"""Heads on cached X3D-S features. Every head takes (B, T, 2048).

Stage 1 (binary MIL) heads return per-clip scores in [0, 1], shape (B, T).
Stage 2 (14-class) heads return video logits, shape (B, 14).

T is the number of clips: 32 segments in training, the last few clips on the Pi.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import spec


class BinaryMLP(nn.Module):
    """Per-clip MLP (Sultani et al.): 2048 -> 512 -> 32 -> 1, sigmoid."""

    def __init__(self, in_dim=spec.FEATURE_DIM, dropout=0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))


class BinaryTConv(nn.Module):
    """Temporal conv over neighbouring clips, sigmoid per clip."""

    def __init__(self, in_dim=spec.FEATURE_DIM, dropout=0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 512, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(512, 128, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(128, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x.transpose(1, 2)).squeeze(1))


class LinearCls(nn.Module):
    """Mean over clips, then one linear layer (cheapest head; default)."""

    def __init__(self, in_dim=spec.FEATURE_DIM, n_classes=len(spec.CLASSES_14), dropout=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(self.drop(x.mean(dim=1)))


def _topk_mean(logits, k):
    k = max(1, min(k, logits.shape[1]))
    return logits.topk(k, dim=1).values.mean(dim=1)


class MILTopKCls(nn.Module):
    """Per-clip linear logits, video logit = mean of the top-k clips per class."""

    def __init__(self, in_dim=spec.FEATURE_DIM, n_classes=len(spec.CLASSES_14), k=3, dropout=0.5):
        super().__init__()
        self.k = k
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return _topk_mean(self.fc(self.drop(x)), self.k)


class TConvMILCls(nn.Module):
    """Temporal conv per clip, then top-k MIL pooling."""

    def __init__(self, in_dim=spec.FEATURE_DIM, n_classes=len(spec.CLASSES_14), k=3, dropout=0.5):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Conv1d(in_dim, 256, 3, padding=1), nn.ReLU(),
            nn.Dropout(dropout), nn.Conv1d(256, n_classes, 1),
        )

    def forward(self, x):
        return _topk_mean(self.net(x.transpose(1, 2)).transpose(1, 2), self.k)


BINARY_HEADS = {"mlp": BinaryMLP, "tconv": BinaryTConv}
CLS_HEADS = {"linear": LinearCls, "mil_topk": MILTopKCls, "tconv_mil": TConvMILCls}


def mil_ranking_loss(scores_anom, scores_norm, k=3, lambda_smooth=8e-5, lambda_sparse=8e-5):
    """Top-k ranking hinge + temporal smoothness + sparsity (on anomaly bags).

    scores_anom, scores_norm: (B, T) clip scores in [0, 1].
    """
    k = max(1, min(k, scores_anom.shape[1]))
    top_a = scores_anom.topk(k, dim=1).values.mean(dim=1)
    top_n = scores_norm.topk(k, dim=1).values.mean(dim=1)
    hinge = F.relu(1.0 - top_a + top_n).mean()
    smooth = ((scores_anom[:, 1:] - scores_anom[:, :-1]) ** 2).sum(dim=1).mean()
    sparse = scores_anom.sum(dim=1).mean()
    return hinge + lambda_smooth * smooth + lambda_sparse * sparse
