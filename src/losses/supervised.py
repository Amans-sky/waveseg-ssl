"""Binary Dice + BCE-with-logits loss."""

from __future__ import annotations

import torch
from torch import nn


class DiceBCEWithLogitsLoss(nn.Module):
    """Stable binary loss, expecting logits and target with shape ``[B,1,D,H,W]``."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, smooth: float = 1e-5) -> None:
        super().__init__()
        self.dice_weight, self.bce_weight, self.smooth = dice_weight, bce_weight, smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(f"Output shape {tuple(logits.shape)} does not match target {tuple(target.shape)}")
        target = target.float()
        probabilities = torch.sigmoid(logits)
        dims = tuple(range(1, logits.ndim))
        dice = (2 * (probabilities * target).sum(dims) + self.smooth) / (probabilities.sum(dims) + target.sum(dims) + self.smooth)
        return self.dice_weight * (1 - dice.mean()) + self.bce_weight * self.bce(logits, target)
