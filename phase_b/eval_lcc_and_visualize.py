"""
WaveSeg-SSL: Plan A Execution
1. Post-Processing Evaluation (3D Largest Connected Component / LCC) on 15 Held-Out Cases
2. Multi-Planar 2D Cross-Sectional Visualizations (Axial, Coronal, Sagittal Overlays)
   for Case 06 (Peak SOTA), Case 07 (High Accuracy), and Case 10 (Outlier Analysis)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label as scipy_label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance

# Paths
DRIVE_ROOT = Path("/content/drive/MyDrive")
DATA_ROOT = DRIVE_ROOT / "WaveSeg-SSL_Datasets/UA-MT/data/2018LA_Seg_Training Set"
SPLIT_PATH = DRIVE_ROOT / "waveseg_ssl_checkpoints/stage3_cps/stage3_cps_split.json"
EXP_DIR = DRIVE_ROOT / "waveseg_ssl_checkpoints/stage4_tuning/exp_wfe_boundary"
BEST_CKPT = EXP_DIR / "best.pt"
VIS_DIR = EXP_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (96, 96, 80)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. Dataset Helpers ---
def resolve_h5(entry: dict) -> Path:
    p = Path(entry.get("h5_path", ""))
    if p.exists(): return p
    case_id = entry.get("case_id", "")
    p_drive = DATA_ROOT / case_id / "mri_norm2.h5"
    if p_drive.exists(): return p_drive
    p_local = Path("/content/UA-MT/data/2018LA_Seg_Training Set") / case_id / "mri_norm2.h5"
    if p_local.exists(): return p_local
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


# --- 2. 3D Haar Wavelet Transform ---
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

# --- 3. Architecture with Bottleneck Hook ---
class WFEBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3, in_channels=1, out_channels=1,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
        )
    def get_bottleneck(self):
        return self.unet.model[1].submodule[1].submodule[1].submodule[1].submodule
    def forward(self, x: torch.Tensor, replacement: Optional[torch.Tensor] = None):
        bottleneck = self.get_bottleneck()
        captured = {}
        def hook(module, inputs, output):
            captured["z"] = output
            return replacement if replacement is not None else output
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

def wfe_forward_single_input(model: WFE_CPS_Model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _, za = model.branch_a(x)
    _, zb = model.branch_b(x)
    ca, sa = haar3d(za)
    cb, sb = haar3d(zb)
    za_wfe = inverse_haar3d((ca[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]), sa)
    zb_wfe = inverse_haar3d((cb[0], ca[1], ca[2], ca[3], ca[4], ca[5], ca[6], ca[7]), sb)
    ya_wfe, _ = model.branch_a(x, replacement=za_wfe)
    yb_wfe, _ = model.branch_b(x, replacement=zb_wfe)
    return ya_wfe, yb_wfe


# --- 4. Post-Processing & Metric Computation ---
def keep_largest_connected_component_3d(binary_mask: np.ndarray) -> np.ndarray:
    """Standard clinical post-processing: keeps only the largest 3D connected component."""
    labeled_array, num_features = scipy_label(binary_mask)
    if num_features <= 1:
        return binary_mask
    bincount = np.bincount(labeled_array.flat)
    bincount[0] = 0  # ignore background
    largest_cc = (labeled_array == bincount.argmax())
    return largest_cc.astype(np.float32)

def compute_metrics_3d(pred_binary: torch.Tensor, target: torch.Tensor):
    dims = tuple(range(1, target.ndim))
    inter = (pred_binary * target).sum(dims)
    denom = pred_binary.sum(dims) + target.sum(dims)
    dice = float(((2.0 * inter + 1e-5) / (denom + 1e-5)).mean().item())

    pred_np = pred_binary[0, 0].cpu().numpy().astype(bool)
    target_np = target[0, 0].cpu().numpy().astype(bool)
    tp = float(np.logical_and(pred_np, target_np).sum())
    fp = float(np.logical_and(pred_np, ~target_np).sum())
    fn = float(np.logical_and(~pred_np, target_np).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    try:
        hd95_t = compute_hausdorff_distance(pred_binary, target, include_background=False, percentile=95.0)
        hd95_val = float(hd95_t[0, 0].cpu().item())
        if not np.isfinite(hd95_val): hd95_val = np.nan
    except Exception:
        hd95_val = np.nan
    return dice, precision, recall, hd95_val


# --- 5. Visualization Engine ---
def save_case_visualization(image_np: np.ndarray, target_np: np.ndarray, pred_raw_np: np.ndarray, pred_lcc_np: np.ndarray, case_id: str, dice_raw: float, hd95_raw: float, dice_lcc: float, hd95_lcc: float):
    """
    Renders 3-view multi-planar cross-sections (Axial, Coronal, Sagittal)
    with Ground Truth (Red), Raw Prediction (Green), and LCC Post-processed Prediction (Cyan).
    """
    # Find center of mass of ground truth target
    fg_indices = np.argwhere(target_np > 0)
    if len(fg_indices) > 0:
        c_z, c_y, c_x = fg_indices.mean(axis=0).astype(int)
    else:
        c_z, c_y, c_x = [s // 2 for s in target_np.shape]

    # Clip to valid bounds
    c_z = int(np.clip(c_z, 0, target_np.shape[0] - 1))
    c_y = int(np.clip(c_y, 0, target_np.shape[1] - 1))
    c_x = int(np.clip(c_x, 0, target_np.shape[2] - 1))

    slices = [
        ("Axial (Z-Slice)", image_np[c_z, :, :], target_np[c_z, :, :], pred_raw_np[c_z, :, :], pred_lcc_np[c_z, :, :]),
        ("Coronal (Y-Slice)", image_np[:, c_y, :], target_np[:, c_y, :], pred_raw_np[:, c_y, :], pred_lcc_np[:, c_y, :]),
        ("Sagittal (X-Slice)", image_np[:, :, c_x], target_np[:, :, c_x], pred_raw_np[:, :, c_x], pred_lcc_np[:, :, c_x]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#1e1e1e")
    fig.suptitle(
        f"Case: {case_id}\nRaw WFE: Dice={dice_raw:.4f}, HD95={hd95_raw:.2f} vox | Post-LCC: Dice={dice_lcc:.4f}, HD95={hd95_lcc:.2f} vox",
        color="white", fontsize=15, fontweight="bold", y=1.02
    )

    for i, (title, im_slice, gt_slice, pr_slice, lcc_slice) in enumerate(slices):
        ax = axes[i]
        ax.imshow(im_slice, cmap="gray", origin="lower")

        # Ground truth in Red
        if gt_slice.max() > 0:
            ax.contour(gt_slice, levels=[0.5], colors=["#FF3366"], linewidths=2.0)
        # Raw prediction in Lime Green (dashed)
        if pr_slice.max() > 0:
            ax.contour(pr_slice, levels=[0.5], colors=["#33FF33"], linewidths=1.5, linestyles="dashed")
        # Post-LCC prediction in Cyan (solid)
        if lcc_slice.max() > 0:
            ax.contour(lcc_slice, levels=[0.5], colors=["#00E5FF"], linewidths=1.8)

        ax.set_title(title, color="white", fontsize=12)
        ax.axis("off")

    legend_elements = [
        mpatches.Patch(color="#FF3366", label="Ground Truth (Manual)"),
        mpatches.Patch(color="#33FF33", label="WFE Prediction (Raw)"),
        mpatches.Patch(color="#00E5FF", label="WFE + LCC (Post-Processed)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, facecolor="#2d2d2d", edgecolor="none", labelcolor="white", fontsize=12)
    plt.tight_layout()

    out_path = VIS_DIR / f"{case_id}_multiview.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED] Visualization: {out_path.name}")


# --- 6. Main Pipeline ---
def run_plan_a():
    print("=" * 80)
    print("WAVESEG-SSL: PLAN A (LCC POST-PROCESSING & MULTI-VIEW VISUALIZATIONS)")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    model = WFE_CPS_Model().to(DEVICE)
    ckpt = torch.load(BEST_CKPT, map_location=DEVICE, weights_only=False)
    model.branch_a.load_state_dict(ckpt["model_a_state"])
    model.branch_b.load_state_dict(ckpt["model_b_state"])
    model.eval()

    with open(SPLIT_PATH, "r") as f:
        test_entries = json.load(f)["splits"]["test"]

    test_loader = DataLoader(FullVolumeTestDataset(test_entries), batch_size=1, shuffle=False)

    def wfe_predictor(patch: torch.Tensor) -> torch.Tensor:
        ya_wfe, yb_wfe = wfe_forward_single_input(model, patch)
        return 0.5 * (ya_wfe + yb_wfe)

    target_cases_for_vis = [
        "VIXBEFTNVHZWKAKURJBN",  # Case 06: Best Peak Case (HD95: 7.81 vox)
        "4EPVTT1HPA8U60CDUKXE",  # Case 07: High Precision (HD95: 14.49 vox)
        "A4R1S23KR0KU2WSYHK2X",  # Case 10: Pathological Outlier Diagnosis
    ]

    raw_metrics_list = []
    lcc_metrics_list = []
    per_case_comparison = []

    print("\nProcessing 15 Held-Out Test Cases...")

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            case_id = batch["case_id"][0]
            image, label = batch["image"].to(DEVICE), batch["label"].to(DEVICE)

            # 1. Raw WFE sliding-window prediction
            logits = sliding_window_inference(image, PATCH_SIZE, 1, wfe_predictor, overlap=0.25, mode="gaussian")
            pred_raw = (torch.sigmoid(logits) >= 0.5).float()

            r_dice, r_prec, r_rec, r_hd = compute_metrics_3d(pred_raw, label)
            raw_metrics_list.append((r_dice, r_prec, r_rec, r_hd))

            # 2. 3D Largest Connected Component (LCC)
            pred_raw_np = pred_raw[0, 0].cpu().numpy().astype(np.float32)
            pred_lcc_np = keep_largest_connected_component_3d(pred_raw_np)
            pred_lcc = torch.from_numpy(pred_lcc_np[None, None, ...]).to(DEVICE)

            l_dice, l_prec, l_rec, l_hd = compute_metrics_3d(pred_lcc, label)
            lcc_metrics_list.append((l_dice, l_prec, l_rec, l_hd))

            per_case_comparison.append({
                "case_id": case_id,
                "raw": {"dice": r_dice, "precision": r_prec, "recall": r_rec, "hd95": r_hd},
                "lcc": {"dice": l_dice, "precision": l_prec, "recall": l_rec, "hd95": l_hd},
            })

            print(
                f"[{idx+1:02d}/15] {case_id:20s} | "
                f"Raw Dice: {r_dice:.4f} -> LCC: {l_dice:.4f} | "
                f"Raw HD95: {r_hd:5.2f} -> LCC HD95: {l_hd:5.2f} vox"
            )

            # 3. Generate Visualizations for Key Reference Cases
            if case_id in target_cases_for_vis:
                im_np = image[0, 0].cpu().numpy()
                lbl_np = label[0, 0].cpu().numpy()
                save_case_visualization(
                    image_np=im_np,
                    target_np=lbl_np,
                    pred_raw_np=pred_raw_np,
                    pred_lcc_np=pred_lcc_np,
                    case_id=case_id,
                    dice_raw=r_dice,
                    hd95_raw=r_hd,
                    dice_lcc=l_dice,
                    hd95_lcc=l_hd,
                )

    # Aggregations
    raw_dice = np.array([m[0] for m in raw_metrics_list])
    raw_prec = np.array([m[1] for m in raw_metrics_list])
    raw_rec = np.array([m[2] for m in raw_metrics_list])
    raw_hd = np.array([m[3] for m in raw_metrics_list if np.isfinite(m[3])])

    lcc_dice = np.array([m[0] for m in lcc_metrics_list])
    lcc_prec = np.array([m[1] for m in lcc_metrics_list])
    lcc_rec = np.array([m[2] for m in lcc_metrics_list])
    lcc_hd = np.array([m[3] for m in lcc_metrics_list if np.isfinite(m[3])])

    summary = {
        "experiment": "stage4_wfe_boundary_plan_a",
        "raw_summary": {
            "mean_dice": float(np.mean(raw_dice)), "mean_precision": float(np.mean(raw_prec)),
            "mean_recall": float(np.mean(raw_rec)), "mean_hd95": float(np.mean(raw_hd)),
        },
        "lcc_summary": {
            "mean_dice": float(np.mean(lcc_dice)), "mean_precision": float(np.mean(lcc_prec)),
            "mean_recall": float(np.mean(lcc_rec)), "mean_hd95": float(np.mean(lcc_hd)),
        },
        "per_case": per_case_comparison,
    }

    out_json = EXP_DIR / "test_metrics_with_lcc.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 85)
    print("PLAN A: RAW VS. LCC POST-PROCESSED COMPARISON (15 TEST VOLUMES)")
    print("=" * 85)
    print(f"{'Metric':<25} | {'Raw WFE + Boundary':<25} | {'Post-Processed (+ 3D LCC)':<25} | {'Impact / Delta'}")
    print("-" * 85)
    print(f"{'Mean Dice':<25} | {summary['raw_summary']['mean_dice']:<25.6f} | {summary['lcc_summary']['mean_dice']:<25.6f} | {summary['lcc_summary']['mean_dice'] - summary['raw_summary']['mean_dice']:+.4f}")
    print(f"{'Mean Precision':<25} | {summary['raw_summary']['mean_precision']:<25.6f} | {summary['lcc_summary']['mean_precision']:<25.6f} | {summary['lcc_summary']['mean_precision'] - summary['raw_summary']['mean_precision']:+.4f}")
    print(f"{'Mean Recall':<25} | {summary['raw_summary']['mean_recall']:<25.6f} | {summary['lcc_summary']['mean_recall']:<25.6f} | {summary['lcc_summary']['mean_recall'] - summary['raw_summary']['mean_recall']:+.4f}")
    print(f"{'Mean HD95 (voxels)':<25} | {summary['raw_summary']['mean_hd95']:<25.4f} | {summary['lcc_summary']['mean_hd95']:<25.4f} | {summary['lcc_summary']['mean_hd95'] - summary['raw_summary']['mean_hd95']:+.2f} vox")
    print("=" * 85)
    print(f"Metrics saved to : {out_json}")
    print(f"Visualizations in: {VIS_DIR}")


if __name__ == "__main__":
    run_plan_a()
