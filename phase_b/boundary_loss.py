"""
WaveSeg-SSL: Boundary-Aware Loss Module
========================================
Implements a differentiable, 3D boundary-aware loss derived from morphological
band extraction for volumetric medical image segmentation (LA MRI).

Mathematical formulation:
-------------------------
For ground-truth binary mask Y in {0, 1}^(B x 1 x D x H x W):
1. Dilation: dilate(Y) = max_pool3d(Y, kernel_size=3, stride=1, padding=1)
2. Erosion:  erode(Y)  = 1.0 - max_pool3d(1.0 - Y, kernel_size=3, stride=1, padding=1)
3. Target boundary band: B_target = dilate(Y) - erode(Y)

For predicted segmentation probabilities P = sigmoid(logits) in [0, 1]^(B x 1 x D x H x W):
1. dilate(P) = max_pool3d(P, kernel_size=3, stride=1, padding=1)
2. erode(P)  = 1.0 - max_pool3d(1.0 - P, kernel_size=3, stride=1, padding=1)
3. Pred boundary representation: B_pred = dilate(P) - erode(P)

Boundary Dice Loss:
    L_bdice(B_pred, B_target) = 1.0 - (2 * sum(B_pred * B_target) + eps) / (sum(B_pred) + sum(B_target) + eps)

Boundary BCE Loss (optional combination):
    L_bbce(logits, B_target) = BCEWithLogits(logits_boundary, B_target)

Properties:
- Fully differentiable via PyTorch native max_pool3d
- Strictly scale-bounded in [0, 1], directly compatible with standard Dice loss
- Numerically stable for all-background (empty) and all-foreground masks
- Robust against division-by-zero errors via epsilon smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_boundary_band_3d(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Extracts a boundary band of given thickness from a 3D volumetric mask/probability map.
    Supports both hard binary masks and continuous probability maps (soft morphology).

    Args:
        mask: Tensor of shape (B, 1, D, H, W) with values in [0, 1].
        kernel_size: Morphological kernel size (default: 3).

    Returns:
        Boundary band tensor of shape (B, 1, D, H, W) in [0, 1].
    """
    pad = kernel_size // 2
    # Dilation via 3D max pooling
    dilated = F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    # Erosion via duality: erode(x) = 1 - dilate(1 - x)
    eroded = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    
    boundary = dilated - eroded
    return boundary.clamp(0.0, 1.0)


class BoundaryDiceLoss(nn.Module):
    """
    Boundary Dice Loss for 3D medical image segmentation.
    Penalizes surface misalignment between predicted boundary and target boundary band.
    """
    def __init__(self, kernel_size: int = 3, smooth: float = 1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits from network, shape (B, 1, D, H, W).
            target: Ground truth binary mask, shape (B, 1, D, H, W).
        Returns:
            Scalar boundary loss in [0, 1].
        """
        probs = torch.sigmoid(logits)
        target = target.float()

        # Differentiable boundary representations
        pred_boundary = extract_boundary_band_3d(probs, kernel_size=self.kernel_size)
        target_boundary = extract_boundary_band_3d(target, kernel_size=self.kernel_size)

        dims = tuple(range(1, logits.ndim))
        intersection = (pred_boundary * target_boundary).sum(dims)
        cardinality = pred_boundary.sum(dims) + target_boundary.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class CombinedBoundarySupervisedLoss(nn.Module):
    """
    Combines standard Supervised Loss (Dice + BCE) with Boundary-Aware Loss:
        L = L_supervised + lambda_boundary * L_boundary
    where:
        L_supervised = Dice_loss + BCE_loss
        L_boundary = Boundary_Dice_loss
    """
    def __init__(self, lambda_boundary: float = 0.1, kernel_size: int = 3, smooth: float = 1e-5):
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.boundary_loss = BoundaryDiceLoss(kernel_size=kernel_size, smooth=smooth)
        self.smooth = smooth

    def dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, logits.ndim))
        intersection = (probs * target).sum(dims)
        cardinality = probs.sum(dims) + target.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        target = target.float()
        
        # Standard supervised components
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = self.dice_loss(logits, target)
        sup_loss = bce + dice

        # Boundary loss component
        b_loss = self.boundary_loss(logits, target)

        total_loss = sup_loss + self.lambda_boundary * b_loss
        return total_loss, sup_loss, b_loss


def run_boundary_loss_unit_tests():
    """
    Runs comprehensive unit tests for numerical stability, edge cases, and gradients.
    """
    print("=" * 70)
    print("RUNNING BOUNDARY LOSS UNIT TESTS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loss_fn = BoundaryDiceLoss().to(device)

    # Test 1: Shape and Finite Output Test
    logits = torch.randn(2, 1, 32, 32, 32, device=device, requires_grad=True)
    target = (torch.rand(2, 1, 32, 32, 32, device=device) > 0.7).float()
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss).all(), "Loss must be finite"
    print(f"[TEST 1] Random input loss: {loss.item():.5f} -> PASS (finite)")

    # Test 2: Gradient Propagation Test
    loss.backward()
    assert logits.grad is not None, "Gradients must exist"
    assert torch.isfinite(logits.grad).all(), "Gradients must be finite"
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad)), "Gradients must be non-zero"
    print("[TEST 2] Gradient propagation: PASS (finite and non-zero)")

    # Test 3: Perfect Prediction Test (should yield near 0 boundary loss)
    target_clean = torch.zeros(1, 1, 32, 32, 32, device=device)
    target_clean[:, :, 10:22, 10:22, 10:22] = 1.0  # Cube in center
    logits_perfect = (target_clean * 20.0) - 10.0    # High positive inside, high negative outside
    loss_perfect = loss_fn(logits_perfect, target_clean)
    assert loss_perfect.item() < 0.05, f"Perfect prediction loss should be near 0, got {loss_perfect.item()}"
    print(f"[TEST 3] Perfect prediction loss: {loss_perfect.item():.5f} -> PASS (< 0.05)")

    # Test 4: Shifted Boundary Test (should yield higher loss than perfect)
    target_shifted = torch.zeros_like(target_clean)
    target_shifted[:, :, 13:25, 13:25, 13:25] = 1.0  # Shifted cube
    loss_shifted = loss_fn(logits_perfect, target_shifted)
    assert loss_shifted.item() > loss_perfect.item(), "Shifted boundary must yield higher loss"
    print(f"[TEST 4] Shifted boundary loss: {loss_shifted.item():.5f} (vs perfect {loss_perfect.item():.5f}) -> PASS")

    # Test 5: Empty Foreground Edge Case (all background, target = 0)
    target_empty = torch.zeros(1, 1, 32, 32, 32, device=device)
    logits_empty = torch.zeros_like(target_empty)
    loss_empty = loss_fn(logits_empty, target_empty)
    assert torch.isfinite(loss_empty), "Empty foreground must not result in NaN/Inf"
    print(f"[TEST 5] Empty foreground loss: {loss_empty.item():.5f} -> PASS (finite)")

    # Test 6: Full Foreground Edge Case (all foreground, target = 1)
    target_full = torch.ones(1, 1, 32, 32, 32, device=device)
    loss_full = loss_fn(logits_empty, target_full)
    assert torch.isfinite(loss_full), "Full foreground must not result in NaN/Inf"
    print(f"[TEST 6] Full foreground loss: {loss_full.item():.5f} -> PASS (finite)")

    print("=" * 70)
    print("ALL BOUNDARY LOSS UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_boundary_loss_unit_tests()
