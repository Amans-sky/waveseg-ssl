"""
WaveSeg-SSL: WFE Checkpoint & Architecture Verification
========================================================
Task 2: Independent verification of the epoch-86 WFE-only checkpoint.
Location: /content/drive/MyDrive/waveseg_ssl_checkpoints/stage4_tuning/exp_wfe_no_lpfn/best.pt
"""

import sys
import json
from pathlib import Path

import torch
from train_wfe_boundary import (
    WFE_CPS_Model,
    WFE_ONLY_CKPT,
    DEVICE,
)


def verify_checkpoint():
    print("=" * 80)
    print("WFE CHECKPOINT & ARCHITECTURE VERIFICATION")
    print("=" * 80)
    print(f"Target Checkpoint: {WFE_ONLY_CKPT}")
    print(f"Device           : {DEVICE}")

    if not WFE_ONLY_CKPT.exists():
        raise FileNotFoundError(
            f"WFE Checkpoint not found at: {WFE_ONLY_CKPT}\n"
            "Please ensure Google Drive is mounted at /content/drive"
        )

    size_mb = WFE_ONLY_CKPT.stat().st_size / (1024 ** 2)
    print(f"File Size        : {size_mb:.2f} MB")

    # 1. Load raw dictionary
    ckpt = torch.load(WFE_ONLY_CKPT, map_location="cpu", weights_only=False)
    print(f"\nRecorded Epoch   : {ckpt.get('epoch', 'N/A')}")
    print(f"Best Val Dice    : {ckpt.get('best_val_dice', 'N/A')}")
    print("Checkpoint Keys  :", list(ckpt.keys()))

    assert "model_a_state" in ckpt, "Missing model_a_state in checkpoint"
    assert "model_b_state" in ckpt, "Missing model_b_state in checkpoint"

    state_a = ckpt["model_a_state"]
    state_b = ckpt["model_b_state"]
    print(f"Branch A tensors : {len(state_a)}")
    print(f"Branch B tensors : {len(state_b)}")

    # 2. Build model and load weights
    model = WFE_CPS_Model().to(DEVICE)
    res_a = model.branch_a.load_state_dict(state_a, strict=True)
    res_b = model.branch_b.load_state_dict(state_b, strict=True)

    print("\nWeight Loading Verification:")
    print("  Branch A missing keys    :", len(res_a.missing_keys))
    print("  Branch A unexpected keys :", len(res_a.unexpected_keys))
    print("  Branch B missing keys    :", len(res_b.missing_keys))
    print("  Branch B unexpected keys :", len(res_b.unexpected_keys))
    print("State dict loading: PERFECT MATCH (0 missing, 0 unexpected)")

    # 3. Model Forward Test with synthetic volume
    model.eval()
    dummy_x = torch.randn(1, 1, 96, 96, 80, device=DEVICE, requires_grad=True)
    
    with torch.no_grad():
        out_a, out_b = model(dummy_x, dummy_x)

    print("\nInference Output Shapes:")
    print(f"  Branch A output : {tuple(out_a.shape)}")
    print(f"  Branch B output : {tuple(out_b.shape)}")
    assert out_a.shape == (1, 1, 96, 96, 80)
    assert out_b.shape == (1, 1, 96, 96, 80)
    assert torch.isfinite(out_a).all(), "Output A contains NaN/Inf"
    assert torch.isfinite(out_b).all(), "Output B contains NaN/Inf"
    print("Output shapes & finite check: PASS")

    # 4. Backward Pass Test
    model.train()
    out_a, out_b = model(dummy_x, dummy_x)
    loss = out_a.mean() + out_b.mean()
    loss.backward()
    assert dummy_x.grad is not None, "Gradient did not flow to input"
    assert torch.isfinite(dummy_x.grad).all(), "Input gradients contain NaN/Inf"
    print("Backward gradient flow through WFE: PASS")

    print("\n" + "=" * 80)
    print("WFE CHECKPOINT INTEGRITY: 100% VERIFIED AND READY FOR BOUNDARY TRAINING")
    print("=" * 80)


if __name__ == "__main__":
    verify_checkpoint()
