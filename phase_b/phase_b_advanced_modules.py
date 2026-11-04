"""
WaveSeg-SSL: Phase B - Advanced Faculty Recommendations
======================================================
1. nnU-Net Style 3D Medical Augmentation Pipeline (Faculty Question 8)
2. Uncertainty-Aware / Entropy-Weighted Dynamic CPS (Faculty Question 6)
3. Foundation Model + LoRA Architectural Blueprint (Faculty Question 7)

Ready to drop into Colab as soon as compute resets.
"""

import math
from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.transforms import (
    Compose,
    RandRotated,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandSpatialCropd,
    RandFlipd,
    EnsureChannelFirstd,
)


# ==============================================================================
# 1. nnU-Net STYLE 3D AUGMENTATION PIPELINE (FACULTY QUESTION 8)
# ==============================================================================

def get_nnunet_style_3d_augmentations(patch_size: Tuple[int, int, int] = (96, 96, 80)):
    """
    Implements the core stochastic spatial and intensity augmentations
    from the nnU-Net paper (Isensee et al., Nature Methods 2021):
    - Random 3D Rotation (within +/- 30 degrees)
    - Random 3D Spatial Scaling (0.7x to 1.4x)
    - Additive Gaussian Noise (N(0, 0.1))
    - Gaussian Blur / Smoothing (sigma 0.5 to 1.0)
    - Gamma Contrast Transform (gamma 0.7 to 1.5)
    - Random 3D Axis Flips
    """
    return Compose([
        # Random 3D Spatial Rotation (+/- 30 deg on all 3 axes)
        RandRotated(
            keys=["image", "label"],
            range_x=(-np.pi / 6, np.pi / 6),
            range_y=(-np.pi / 6, np.pi / 6),
            range_z=(-np.pi / 6, np.pi / 6),
            mode=("bilinear", "nearest"),
            prob=0.3,
            padding_mode="reflection",
        ),
        # Random Axis Flips (x, y, z)
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
        # Additive Gaussian Noise
        RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.1),
        # Gaussian Blurring
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        # Gamma Contrast Adjustment
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
    ])


# ==============================================================================
# 2. UNCERTAINTY-AWARE DYNAMIC CPS LOSS (FACULTY QUESTION 6)
# ==============================================================================

def compute_voxelwise_entropy(probs: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Shannon entropy for binary predictions:
    H(p) = - [p * log(p) + (1-p) * log(1-p)] / log(2)
    Normalized to [0.0, 1.0]. High entropy = high model uncertainty.
    """
    probs = probs.clamp(eps, 1.0 - eps)
    entropy = -(probs * torch.log2(probs) + (1.0 - probs) * torch.log2(1.0 - probs))
    return entropy


def uncertainty_aware_cps_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    entropy_threshold: float = 0.5,
    confidence_threshold: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replaces static hard thresholding with dual Uncertainty-Confidence gating:
    1. Confidence Gating: max(p, 1-p) >= confidence_threshold
    2. Uncertainty Filtering: Shannon Entropy H(p) <= entropy_threshold
    
    Voxels where the teacher is uncertain (e.g. blurry myocardial borders)
    are discounted, preventing confirmation bias in CPS.
    """
    teacher_prob = torch.sigmoid(teacher_logits).detach()
    pseudo_labels = (teacher_prob >= 0.5).float()

    # Voxel-wise confidence and entropy
    confidence = torch.maximum(teacher_prob, 1.0 - teacher_prob)
    entropy = compute_voxelwise_entropy(teacher_prob)

    # Uncertainty-aware valid mask: High Confidence AND Low Entropy
    valid_mask = ((confidence >= confidence_threshold) & (entropy <= entropy_threshold)).float()

    # Uncertainty-weighted BCE
    bce = F.binary_cross_entropy_with_logits(student_logits, pseudo_labels, reduction="none")
    weighted_bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return weighted_bce, valid_mask.mean()


# ==============================================================================
# 3. FOUNDATION MODEL + LoRA BLUEPRINT (FACULTY QUESTION 7)
# ==============================================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for Transformer Attention Projections:
    W = W_0 + (alpha / r) * (B @ A)
    Where W_0 is frozen, A is Gaussian initialized, B is zero initialized.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Trainable low-rank adapters
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight)
        lora_out = F.linear(x, self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling
        return base_out + lora_out


print("Phase B Modules loaded successfully: nnU-Net Augmentations, Uncertainty CPS, and LoRA Blueprint.")
