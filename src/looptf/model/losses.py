"""Next-token cross-entropy on the soft-capped logits."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def token_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Per-token cross-entropy with the shape of `labels`, computed in FP32; positions labelled -100 are zero."""
    ce = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none",
    ).view_as(labels)
    return ce.masked_fill(labels.eq(-100), 0.0)


def next_token_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean cross-entropy of predicting labels[:, 1:] from logits[:, :-1], skipping labels of -100."""
    targets = labels[:, 1:]
    return token_cross_entropy(logits[:, :-1], targets).sum() / targets.ne(-100).sum().clamp_min(1)
