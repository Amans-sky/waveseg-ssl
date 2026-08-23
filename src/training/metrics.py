from __future__ import annotations
import torch

def binary_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> torch.Tensor:
    prediction = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    dims = tuple(range(1, logits.ndim))
    return ((2 * (prediction * target).sum(dims) + smooth) / (prediction.sum(dims) + target.sum(dims) + smooth)).mean()
