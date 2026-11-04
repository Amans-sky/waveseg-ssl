"""
WaveSeg-SSL: Frozen Locked Test Set Evaluation
==============================================
CRITICAL PROTOCOL GUARD:
This script must ONLY be executed AFTER the final model configuration is frozen.
DO NOT use test metrics for tuning, model selection, loss-weight selection, or early stopping.

Evaluates the 15 locked test cases from:
/content/drive/MyDrive/waveseg_ssl_checkpoints/stage3_cps/stage3_cps_split.json

Calculates:
- Mean, Median, Std, Min, Max Dice
- Precision & Recall
- HD95 (95% Hausdorff Distance)
- Dual Evaluation:
  1) Genuine WFE Ensemble (Haar frequency exchange at test time)
  2) Standard Plain UNet Ensemble (for direct baseline comparison)
- Per-case detailed breakdown

Persists results to:
/content/drive/MyDrive/waveseg_ssl_checkpoints/stage4_tuning/exp_wfe_boundary/test_metrics_detailed.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance

# Paths (Colab environment)
DRIVE_ROOT = Path("/content/drive/MyDrive")
DATA_ROOT = DRIVE_ROOT / "WaveSeg-SSL_Datasets/UA-MT/data/2018LA_Seg_Training Set"
SPLIT_PATH = DRIVE_ROOT / "waveseg_ssl_checkpoints/stage3_cps/stage3_cps_split.json"
EXP_DIR = DRIVE_ROOT / "waveseg_ssl_checkpoints/stage4_tuning/exp_wfe_boundary"
BEST_CKPT = EXP_DIR / "best.pt"

PATCH_SIZE = (96, 96, 80)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Safety lock: must be set to True explicitly
CONFIRM_FINAL_CONFIGURATION_FROZEN = True


# ============================================================
# RESOLUTION & DATASET
# ============================================================

def resolve_h5(entry: dict) -> Path:
    p = Path(entry.get("h5_path", ""))
    if p.exists():
        return p
    case_id = entry.get("case_id", "")
    p_drive = DATA_ROOT / case_id / "mri_norm2.h5"
    if p_drive.exists():
        return p_drive
    p_local = Path("/content/UA-MT/data/2018LA_Seg_Training Set") / case_id / "mri_norm2.h5"
    if p_local.exists():
        return p_local
    raise FileNotFoundError(f"Could not locate mri_norm2.h5 for case {case_id}")


class FullVolumeTestDataset(Dataset):
    def __init__(self, entries: List[dict]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        path = resolve_h5(entry)
        with h5py.File(path, "r") as handle:
            image = np.asarray(handle["image"], dtype=np.float32)
            label = np.asarray(handle["label"], dtype=np.uint8)

        return {
            "image": torch.from_numpy(image[None, ...]).float(),
            "label": torch.from_numpy(label[None, ...]).float(),
            "case_id": entry.get("case_id", f"case_{idx}"),
        }


# ============================================================
# 3D HAAR WAVELET FREQUENCY EXCHANGE (WFE)
# ============================================================

def haar3d(x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], Tuple[int, int, int]]:
    orig = x.shape[-3:]
    pd, ph, pw = orig[0] % 2, orig[1] % 2, orig[2] % 2
    if pd or ph or pw:
        x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="reflect")

    x000, x001 = x[..., 0::2, 0::2, 0::2], x[..., 0::2, 0::2, 1::2]
    x010, x011 = x[..., 0::2, 1::2, 0::2], x[..., 0::2, 1::2, 1::2]
    x100, x101 = x[..., 1::2, 0::2, 0::2], x[..., 1::2, 0::2, 1::2]
    x110, x111 = x[..., 1::2, 1::2, 0::2], x[..., 1::2, 1::2, 1::2]

    s = 1.0 / (2.0 ** 1.5)
    return (
        ( x000 + x001 + x010 + x011 + x100 + x101 + x110 + x111) * s,  # LLL
        (-x000 - x001 - x010 - x011 + x100 + x101 + x110 + x111) * s,  # H1
        (-x000 - x001 + x010 + x011 - x100 - x101 + x110 + x111) * s,  # H2
        (-x000 - x001 + x010 + x011 + x100 + x101 - x110 - x111) * s,  # H3
        (-x000 + x001 - x010 + x011 - x100 + x101 - x110 + x111) * s,  # H4
        (-x000 + x001 - x010 + x011 + x100 - x101 + x110 - x111) * s,  # H5
        (-x000 + x001 + x010 - x011 - x100 + x101 + x110 - x111) * s,  # H6
        (-x000 + x001 + x010 - x011 + x100 - x101 - x110 + x111) * s,  # H7
    ), orig


def inverse_haar3d(coeffs: Tuple[torch.Tensor, ...], orig: Tuple[int, int, int]) -> torch.Tensor:
    ll, h1, h2, h3, h4, h5, h6, h7 = coeffs
    s = 1.0 / (2.0 ** 1.5)

    x000 = (ll - h1 - h2 - h3 - h4 - h5 - h6 - h7) * s
    x001 = (ll - h1 - h2 - h3 + h4 + h5 + h6 + h7) * s
    x010 = (ll - h1 + h2 + h3 - h4 - h5 + h6 + h7) * s
    x011 = (ll - h1 + h2 + h3 + h4 + h5 - h6 - h7) * s
    x100 = (ll + h1 - h2 + h3 - h4 + h5 - h6 + h7) * s
    x101 = (ll + h1 - h2 + h3 + h4 - h5 + h6 - h7) * s
    x110 = (ll + h1 + h2 - h3 - h4 + h5 + h6 - h7) * s
    x111 = (ll + h1 + h2 - h3 + h4 - h5 - h6 + h7) * s

    B, C, D, H, W = ll.shape
    out = torch.zeros(B, C, D * 2, H * 2, W * 2, device=ll.device, dtype=ll.dtype)
    out[..., 0::2, 0::2, 0::2] = x000
    out[..., 0::2, 0::2, 1::2] = x001
    out[..., 0::2, 1::2, 0::2] = x010
    out[..., 0::2, 1::2, 1::2] = x011
    out[..., 1::2, 0::2, 0::2] = x100
    out[..., 1::2, 0::2, 1::2] = x101
    out[..., 1::2, 1::2, 0::2] = x110
    out[..., 1::2, 1::2, 1::2] = x111
    return out[..., :orig[0], :orig[1], :orig[2]]


# ============================================================
# ARCHITECTURE (STAGE-3 MONAI UNET + BOTTLENECK HOOK)
# ============================================================

class WFEBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def get_bottleneck(self):
        return (
            self.unet.model[1]
            .submodule[1]
            .submodule[1]
            .submodule[1]
            .submodule
        )

    def forward(self, x: torch.Tensor, replacement: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.get_bottleneck()
        captured = {}

        def hook(module, inputs, output):
            captured["z"] = output
            if replacement is not None:
                return replacement
            return output

        handle = bottleneck.register_forward_hook(hook)
        try:
            y = self.unet(x)
        finally:
            handle.remove()

        return y, captured["z"]


class WFE_CPS_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = WFEBranch()
        self.branch_b = WFEBranch()

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, z_a = self.branch_a(x_a)
        _, z_b = self.branch_b(x_b)
        ca, sa = haar3d(z_a)
        cb, sb = haar3d(z_b)
        z_a_wfe = inverse_haar3d((ca[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]), sa)
        z_b_wfe = inverse_haar3d((cb[0], ca[1], ca[2], ca[3], ca[4], ca[5], ca[6], ca[7]), sb)
        y_a, _ = self.branch_a(x_a, replacement=z_a_wfe)
        y_b, _ = self.branch_b(x_b, replacement=z_b_wfe)
        return y_a, y_b


def wfe_forward_single_input(model: WFE_CPS_Model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Genuine WFE inference: captures za, zb, swaps HF subbands, decodes replaced representations."""
    _, za = model.branch_a(x)
    _, zb = model.branch_b(x)

    ca, sa = haar3d(za)
    cb, sb = haar3d(zb)

    za_wfe = inverse_haar3d((ca[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]), sa)
    zb_wfe = inverse_haar3d((cb[0], ca[1], ca[2], ca[3], ca[4], ca[5], ca[6], ca[7]), sb)

    ya_wfe, _ = model.branch_a(x, replacement=za_wfe)
    yb_wfe, _ = model.branch_b(x, replacement=zb_wfe)
    return ya_wfe, yb_wfe


# ============================================================
# EVALUATION METRICS ENGINE
# ============================================================

def compute_metrics_3d(pred_binary: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, float]:
    """Calculates Dice, Precision, Recall, and HD95 (voxels) for a single 3D case."""
    # Binary Dice
    dims = tuple(range(1, target.ndim))
    inter = (pred_binary * target).sum(dims)
    denom = pred_binary.sum(dims) + target.sum(dims)
    dice = float(((2.0 * inter + 1e-5) / (denom + 1e-5)).mean().item())

    # Precision & Recall
    pred_np = pred_binary[0, 0].cpu().numpy().astype(bool)
    target_np = target[0, 0].cpu().numpy().astype(bool)

    tp = float(np.logical_and(pred_np, target_np).sum())
    fp = float(np.logical_and(pred_np, ~target_np).sum())
    fn = float(np.logical_and(~pred_np, target_np).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # HD95
    try:
        hd95_t = compute_hausdorff_distance(pred_binary, target, include_background=False, percentile=95.0)
        hd95_val = float(hd95_t[0, 0].cpu().item())
        if not np.isfinite(hd95_val):
            hd95_val = np.nan
    except Exception:
        hd95_val = np.nan

    return dice, precision, recall, hd95_val


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def evaluate_locked_test_set():
    print("=" * 80)
    print("WAVESEG-SSL: LOCKED TEST SET EVALUATION (FROZEN CONFIGURATION)")
    print("=" * 80)

    if not CONFIRM_FINAL_CONFIGURATION_FROZEN:
        raise RuntimeError(
            "SAFETY GUARD BLOCKED: The test set is locked and must remain untouched\n"
            "until the model configuration and training run are completely frozen.\n"
            "To execute, set CONFIRM_FINAL_CONFIGURATION_FROZEN = True."
        )

    assert BEST_CKPT.exists(), f"Best checkpoint not found at: {BEST_CKPT}"
    assert SPLIT_PATH.exists(), f"Split file not found at: {SPLIT_PATH}"

    print(f"Loading Frozen Checkpoint : {BEST_CKPT}")
    print(f"Device                    : {DEVICE}")

    # 1. Load Model & Weights
    model = WFE_CPS_Model().to(DEVICE)
    ckpt = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)

    model.branch_a.load_state_dict(ckpt["model_a_state"])
    model.branch_b.load_state_dict(ckpt["model_b_state"])
    model.eval()

    best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", "N/A"))
    best_val_dice = ckpt.get("best_val_dice", "N/A")
    print(f"Model Best Epoch          : {best_epoch}")
    print(f"Model Best Val Dice       : {best_val_dice:.6f}" if isinstance(best_val_dice, float) else f"Model Best Val Dice       : {best_val_dice}")

    # 2. Load Locked Split
    with open(SPLIT_PATH, "r") as f:
        split_data = json.load(f)["splits"]

    test_entries = split_data["test"]
    assert len(test_entries) == 15, f"Expected 15 test cases, found {len(test_entries)}"
    print(f"Locked Test Cases Count   : {len(test_entries)}")

    test_ds = FullVolumeTestDataset(test_entries)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Storage for WFE Genuine Inference
    wfe_dice_s, wfe_prec_s, wfe_rec_s, wfe_hd95_s = [], [], [], []
    case_results = []

    # Storage for Standard Inference (without test-time WFE hook)
    std_dice_s, std_prec_s, std_rec_s, std_hd95_s = [], [], [], []

    print("\nRunning Full-Volume Sliding-Window Inference on Held-Out Test Set (15 cases)...\n")

    # Predictor with genuine WFE
    def wfe_predictor(patch: torch.Tensor) -> torch.Tensor:
        ya_wfe, yb_wfe = wfe_forward_single_input(model, patch)
        return 0.5 * (ya_wfe + yb_wfe)

    # Standard predictor (direct branch averaging)
    def std_predictor(patch: torch.Tensor) -> torch.Tensor:
        ya, _ = model.branch_a(patch)
        yb, _ = model.branch_b(patch)
        return 0.5 * (ya + yb)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            case_id = batch["case_id"][0]
            image = batch["image"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            # 1. Genuine WFE Sliding Window Inference
            logits_wfe = sliding_window_inference(
                image,
                roi_size=PATCH_SIZE,
                sw_batch_size=1,
                predictor=wfe_predictor,
                overlap=0.25,
                mode="gaussian",
            )
            prob_wfe = torch.sigmoid(logits_wfe)
            pred_wfe = (prob_wfe >= 0.5).float()

            w_dice, w_prec, w_rec, w_hd95 = compute_metrics_3d(pred_wfe, label)
            wfe_dice_s.append(w_dice)
            wfe_prec_s.append(w_prec)
            wfe_rec_s.append(w_rec)
            wfe_hd95_s.append(w_hd95)

            # 2. Standard Plain Sliding Window Inference
            logits_std = sliding_window_inference(
                image,
                roi_size=PATCH_SIZE,
                sw_batch_size=1,
                predictor=std_predictor,
                overlap=0.25,
                mode="gaussian",
            )
            prob_std = torch.sigmoid(logits_std)
            pred_std = (prob_std >= 0.5).float()

            s_dice, s_prec, s_rec, s_hd95 = compute_metrics_3d(pred_std, label)
            std_dice_s.append(s_dice)
            std_prec_s.append(s_prec)
            std_rec_s.append(s_rec)
            std_hd95_s.append(s_hd95)

            case_res = {
                "case_id": case_id,
                "wfe_inference": {
                    "dice": w_dice,
                    "precision": w_prec,
                    "recall": w_rec,
                    "hd95": w_hd95,
                },
                "standard_inference": {
                    "dice": s_dice,
                    "precision": s_prec,
                    "recall": s_rec,
                    "hd95": s_hd95,
                },
            }
            case_results.append(case_res)

            print(
                f"[{idx+1:02d}/15] {case_id:20s} | "
                f"WFE Dice: {w_dice:.4f} | Prec: {w_prec:.4f} | Rec: {w_rec:.4f} | HD95: {w_hd95:5.2f} vox || "
                f"Std Dice: {s_dice:.4f} | HD95: {s_hd95:5.2f} vox"
            )

    # Aggregations for WFE Genuine Inference
    wfe_dice_arr = np.asarray(wfe_dice_s)
    wfe_prec_arr = np.asarray(wfe_prec_s)
    wfe_rec_arr = np.asarray(wfe_rec_s)
    wfe_hd95_arr = np.asarray(wfe_hd95_s)
    wfe_valid_hd95 = wfe_hd95_arr[np.isfinite(wfe_hd95_arr)]

    # Aggregations for Standard Inference
    std_dice_arr = np.asarray(std_dice_s)
    std_prec_arr = np.asarray(std_prec_s)
    std_rec_arr = np.asarray(std_rec_s)
    std_hd95_arr = np.asarray(std_hd95_s)
    std_valid_hd95 = std_hd95_arr[np.isfinite(std_hd95_arr)]

    summary = {
        "experiment": "stage4_wfe_boundary",
        "checkpoint": str(BEST_CKPT),
        "best_epoch": int(best_epoch) if str(best_epoch).isdigit() else best_epoch,
        "best_val_dice": float(best_val_dice) if str(best_val_dice).replace('.', '', 1).isdigit() else best_val_dice,
        "test_cases": len(case_results),
        "wfe_genuine_inference": {
            "mean_dice": float(np.mean(wfe_dice_arr)),
            "median_dice": float(np.median(wfe_dice_arr)),
            "std_dice": float(np.std(wfe_dice_arr)),
            "min_dice": float(np.min(wfe_dice_arr)),
            "max_dice": float(np.max(wfe_dice_arr)),
            "mean_precision": float(np.mean(wfe_prec_arr)),
            "median_precision": float(np.median(wfe_prec_arr)),
            "mean_recall": float(np.mean(wfe_rec_arr)),
            "median_recall": float(np.median(wfe_rec_arr)),
            "mean_hd95": float(np.mean(wfe_valid_hd95)) if len(wfe_valid_hd95) > 0 else None,
            "median_hd95": float(np.median(wfe_valid_hd95)) if len(wfe_valid_hd95) > 0 else None,
        },
        "standard_inference": {
            "mean_dice": float(np.mean(std_dice_arr)),
            "median_dice": float(np.median(std_dice_arr)),
            "std_dice": float(np.std(std_dice_arr)),
            "min_dice": float(np.min(std_dice_arr)),
            "max_dice": float(np.max(std_dice_arr)),
            "mean_precision": float(np.mean(std_prec_arr)),
            "median_precision": float(np.median(std_prec_arr)),
            "mean_recall": float(np.mean(std_rec_arr)),
            "median_recall": float(np.median(std_rec_arr)),
            "mean_hd95": float(np.mean(std_valid_hd95)) if len(std_valid_hd95) > 0 else None,
            "median_hd95": float(np.median(std_valid_hd95)) if len(std_valid_hd95) > 0 else None,
        },
        "per_case_metrics": case_results,
    }

    out_metrics_path = EXP_DIR / "test_metrics_detailed.json"
    with open(out_metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    wfe_stats = summary["wfe_genuine_inference"]
    std_stats = summary["standard_inference"]

    print("\n" + "=" * 80)
    print("FINAL LOCKED TEST SET RESULTS (FROZEN EVALUATION)")
    print("=" * 80)
    print("1. GENUINE WFE INFERENCE (Haar Bottleneck Exchange):")
    print(f"   Mean Dice       : {wfe_stats['mean_dice']:.6f}  (Median: {wfe_stats['median_dice']:.6f}, Std: {wfe_stats['std_dice']:.6f})")
    print(f"   Min / Max Dice  : {wfe_stats['min_dice']:.6f} / {wfe_stats['max_dice']:.6f}")
    print(f"   Mean Precision  : {wfe_stats['mean_precision']:.6f}")
    print(f"   Mean Recall     : {wfe_stats['mean_recall']:.6f}")
    print(f"   Mean HD95       : {wfe_stats['mean_hd95']:.4f} voxels" if wfe_stats['mean_hd95'] else "   Mean HD95       : N/A")
    print("\n2. STANDARD PLAIN INFERENCE (Direct Branch Averaging):")
    print(f"   Mean Dice       : {std_stats['mean_dice']:.6f}  (Median: {std_stats['median_dice']:.6f}, Std: {std_stats['std_dice']:.6f})")
    print(f"   Mean Precision  : {std_stats['mean_precision']:.6f}")
    print(f"   Mean Recall     : {std_stats['mean_recall']:.6f}")
    print(f"   Mean HD95       : {std_stats['mean_hd95']:.4f} voxels" if std_stats['mean_hd95'] else "   Mean HD95       : N/A")
    print(f"\nDetailed Metrics Saved To: {out_metrics_path}")
    print("=" * 80)

    # Historical Comparison Table
    print("\n" + "=" * 85)
    print("THREE-STAGE CONTROLLED COMPARISON TABLE (LOCKED TEST SET)")
    print("=" * 85)
    print(f"{'Method / Experiment':<35} | {'Mean Dice':<10} | {'Precision':<10} | {'Recall':<10} | {'HD95 (vox)':<10}")
    print("-" * 85)
    print(f"{'1. Stage 3 (CPS Baseline)':<35} | {'0.676942':<10} | {'0.572284':<10} | {'0.861880':<10} | {'39.043':<10}")
    print(f"{'2. Stage 4 (WFE-Only Baseline)':<35} | {'0.764592':<10} | {'0.666570':<10} | {'0.930268':<10} | {'34.882':<10}")
    print(f"{'3. Stage 4 (WFE + Boundary) [WFE]':<35} | {wfe_stats['mean_dice']:<10.6f} | {wfe_stats['mean_precision']:<10.6f} | {wfe_stats['mean_recall']:<10.6f} | {wfe_stats['mean_hd95'] if wfe_stats['mean_hd95'] else 'N/A':<10.3f}")
    print(f"{'4. Stage 4 (WFE + Boundary) [Std]':<35} | {std_stats['mean_dice']:<10.6f} | {std_stats['mean_precision']:<10.6f} | {std_stats['mean_recall']:<10.6f} | {std_stats['mean_hd95'] if std_stats['mean_hd95'] else 'N/A':<10.3f}")
    print("=" * 85)


if __name__ == "__main__":
    evaluate_locked_test_set()
