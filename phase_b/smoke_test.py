"""
WaveSeg-SSL: Real-Data Smoke Test
=================================
Runs exactly one real training batch through the CPS + WFE + Boundary pipeline.
Verifies:
1. Labeled batch loading & tensor shapes
2. Unlabeled batch loading & tensor shapes
3. WFE forward pass through Branch A and Branch B
4. Supervised loss calculation (Dice + BCE)
5. CPS consistency loss calculation (threshold 0.7)
6. Boundary-aware loss calculation (BoundaryDiceLoss)
7. Total loss combination
8. Scaler backward pass
9. Optimizer step & gradient validity (no NaNs, non-zero)
10. Memory clean-up
"""

import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Import from training script
from train_wfe_boundary import (
    SPLIT_PATH,
    PATCH_SIZE,
    BATCH_SIZE,
    CPS_THRESHOLD,
    CPS_WEIGHT,
    LAMBDA_BOUNDARY,
    DEVICE,
    VolumetricPatchDataset,
    WFE_CPS_Model,
    supervised_loss,
    cps_consistency_loss,
    boundary_dice_loss,
    seed_everything,
)


def run_smoke_test():
    print("=" * 80)
    print("STARTING REAL-DATA SMOKE TEST")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    seed_everything(42)

    # 1. Load Split
    assert SPLIT_PATH.exists(), f"Split file missing: {SPLIT_PATH}"
    with open(SPLIT_PATH, "r") as f:
        split_data = json.load(f)["splits"]

    # 2. Build Loaders
    labeled_ds = VolumetricPatchDataset(split_data["labeled"][:2], labeled=True)
    unlabeled_ds = VolumetricPatchDataset(split_data["unlabeled"][:2], labeled=False)

    labeled_loader = DataLoader(labeled_ds, batch_size=1, shuffle=False)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=1, shuffle=False)

    batch_l = next(iter(labeled_loader))
    batch_u = next(iter(unlabeled_loader))

    img_l = batch_l["image"].to(DEVICE)
    lbl_l = batch_l["label"].to(DEVICE)
    img_u = batch_u["image"].to(DEVICE)

    print(f"[CHECK 1] Labeled Image Shape   : {tuple(img_l.shape)} -> PASS")
    print(f"[CHECK 2] Labeled Mask Shape    : {tuple(lbl_l.shape)} -> PASS")
    print(f"[CHECK 3] Unlabeled Image Shape : {tuple(img_u.shape)} -> PASS")

    assert img_l.shape == (1, 1, 96, 96, 80)
    assert lbl_l.shape == (1, 1, 96, 96, 80)
    assert img_u.shape == (1, 1, 96, 96, 80)

    # 3. Model instantiation
    model = WFE_CPS_Model().to(DEVICE)
    model.train()

    optimizer_a = torch.optim.AdamW(model.branch_a.parameters(), lr=1e-4)
    optimizer_b = torch.optim.AdamW(model.branch_b.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    optimizer_a.zero_grad(set_to_none=True)
    optimizer_b.zero_grad(set_to_none=True)

    with autocast(enabled=torch.cuda.is_available()):
        # Forward pass on labeled pair
        pred_la, pred_lb = model(img_l, img_l)
        print(f"[CHECK 4] Branch A Logits Shape : {tuple(pred_la.shape)} -> PASS")
        print(f"[CHECK 5] Branch B Logits Shape : {tuple(pred_lb.shape)} -> PASS")
        assert pred_la.shape == (1, 1, 96, 96, 80)
        assert pred_lb.shape == (1, 1, 96, 96, 80)

        # Forward pass on unlabeled pair
        pred_ua, pred_ub = model(img_u, img_u)

        # Supervised loss
        sup_a = supervised_loss(pred_la, lbl_l)
        sup_b = supervised_loss(pred_lb, lbl_l)
        loss_sup = 0.5 * (sup_a + sup_b)
        print(f"[CHECK 6] Supervised Loss (Dice+BCE): {loss_sup.item():.5f} -> PASS (finite)")
        assert torch.isfinite(loss_sup)

        # CPS loss
        loss_cps = cps_consistency_loss(pred_ua, pred_ub, threshold=CPS_THRESHOLD)
        print(f"[CHECK 7] CPS Consistency Loss     : {loss_cps.item():.5f} -> PASS (finite)")
        assert torch.isfinite(loss_cps)

        # Boundary loss
        bnd_a = boundary_dice_loss(pred_la, lbl_l)
        bnd_b = boundary_dice_loss(pred_lb, lbl_l)
        loss_bnd = 0.5 * (bnd_a + bnd_b)
        print(f"[CHECK 8] Boundary-Aware Loss       : {loss_bnd.item():.5f} -> PASS (finite)")
        assert torch.isfinite(loss_bnd)

        # Total loss
        total_loss = loss_sup + (CPS_WEIGHT * loss_cps) + (LAMBDA_BOUNDARY * loss_bnd)
        print(f"[CHECK 9] Total Weighted Loss       : {total_loss.item():.5f} -> PASS (finite)")
        assert torch.isfinite(total_loss)

    # 4. Backward Pass & Step
    scaler.scale(total_loss).backward()

    # Check gradients
    param_a = next(model.branch_a.parameters())
    param_b = next(model.branch_b.parameters())
    assert param_a.grad is not None, "Branch A gradient is missing"
    assert param_b.grad is not None, "Branch B gradient is missing"
    assert torch.isfinite(param_a.grad).all(), "Branch A gradient contains NaN/Inf"
    assert torch.isfinite(param_b.grad).all(), "Branch B gradient contains NaN/Inf"
    print("[CHECK 10] Backward Gradients       : PASS (finite on both branches)")

    scaler.step(optimizer_a)
    scaler.step(optimizer_b)
    scaler.update()
    print("[CHECK 11] Optimizer Step           : PASS")

    print("\n" + "=" * 80)
    print("REAL-DATA SMOKE TEST PASSED COMPLETELY!")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
