"""
WaveSeg-SSL: Stage 4 Primary Ablation Training Script
Ablation: CPS + WFE vs CPS + WFE + Boundary-Aware Loss
Strict parity with Stage 4 exp_wfe_no_lpfn baseline protocol with genuine WFE inference for validation.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference


# ============================================================
# CONFIGURATION & REPRODUCIBILITY
# ============================================================

SEED = 42
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (Colab environment)
DRIVE_ROOT = Path("/content/drive/MyDrive")
DATA_ROOT = DRIVE_ROOT / "WaveSeg-SSL_Datasets/UA-MT/data/2018LA_Seg_Training Set"
SPLIT_PATH = DRIVE_ROOT / "waveseg_ssl_checkpoints/stage3_cps/stage3_cps_split.json"
EXP_DIR = DRIVE_ROOT / "waveseg_ssl_checkpoints/stage4_tuning/exp_wfe_boundary"
BEST_PATH = EXP_DIR / "best.pt"
LATEST_PATH = EXP_DIR / "latest.pt"

EXP_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters (Parity with baseline WFE)
PATCH_SIZE = (96, 96, 80)
FOREGROUND_PROBABILITY = 0.75
EPOCHS = 100
LR = 1e-4
CPS_WEIGHT = 1.0
CONFIDENCE_THRESHOLD = 0.7
LAMBDA_BOUNDARY = 0.1
NUM_WORKERS = 2


# ============================================================
# DATASET & RESOLUTION HELPERS
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


def normalize_image(image: np.ndarray, mode: str = "none") -> np.ndarray:
    if mode == "none":
        return image.astype(np.float32, copy=False)
    if mode != "zscore":
        raise ValueError("normalization must be 'none' or 'zscore'.")
    return ((image - image.mean()) / max(float(image.std()), 1e-8)).astype(np.float32, copy=False)


def foreground_or_random_crop(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
    foreground_probability: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if image.shape != label.shape:
        raise ValueError("Image and label must have matching shapes.")
    pads = [
        (max(0, size - cur) // 2, max(0, size - cur) - max(0, size - cur) // 2)
        for cur, size in zip(image.shape, patch_size)
    ]
    image, label = np.pad(image, pads), np.pad(label, pads)
    fg = np.argwhere(label > 0)
    center = (
        fg[rng.integers(len(fg))]
        if len(fg) and rng.random() < foreground_probability
        else np.array([rng.integers(n) for n in image.shape])
    )
    starts = [
        int(np.clip(center[i] - patch_size[i] // 2, 0, image.shape[i] - patch_size[i]))
        for i in range(3)
    ]
    crop = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    return image[crop], label[crop]


class UAMTH5Dataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        *,
        patch_size: Optional[Tuple[int, int, int]] = None,
        foreground_probability: float = 0.75,
        normalization: str = "none",
        seed: int = 42,
    ):
        self.entries = entries
        self.patch_size = patch_size
        self.foreground_probability = foreground_probability
        self.normalization = normalization
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[index]
        path = resolve_h5(entry)
        with h5py.File(path, "r") as handle:
            image = np.asarray(handle["image"], dtype=np.float32)
            label = np.asarray(handle["label"], dtype=np.uint8)

        image = normalize_image(image, self.normalization)
        if self.patch_size:
            image, label = foreground_or_random_crop(
                image, label, self.patch_size, self.foreground_probability, self.rng
            )

        return {
            "image": torch.from_numpy(image[None]).float(),
            "label": torch.from_numpy(label[None].astype(np.float32)),
            "case_id": entry.get("case_id", ""),
        }


def make_loader(entries: List[dict], *, shuffle: bool, seed: int) -> DataLoader:
    ds = UAMTH5Dataset(
        entries,
        patch_size=PATCH_SIZE,
        foreground_probability=FOREGROUND_PROBABILITY,
        normalization="none",
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )


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


def wavelet_frequency_exchange(za: torch.Tensor, zb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ca, sa = haar3d(za)
    cb, sb = haar3d(zb)
    za_wfe = inverse_haar3d((ca[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]), sa)
    zb_wfe = inverse_haar3d((cb[0], ca[1], ca[2], ca[3], ca[4], ca[5], ca[6], ca[7]), sb)
    return za_wfe, zb_wfe


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

        assert z_a.shape[1:] == (256, 6, 6, 5), f"Unexpected bottleneck shape: {z_a.shape}"
        assert z_b.shape[1:] == (256, 6, 6, 5), f"Unexpected bottleneck shape: {z_b.shape}"

        z_a_wfe, z_b_wfe = wavelet_frequency_exchange(z_a, z_b)

        y_a, _ = self.branch_a(x_a, replacement=z_a_wfe)
        y_b, _ = self.branch_b(x_b, replacement=z_b_wfe)

        return y_a, y_b


# ============================================================
# GENUINE WFE INFERENCE FOR VALIDATION
# ============================================================

def wfe_forward_single_input(model: WFE_CPS_Model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Genuine WFE inference for a single validation volume/patch x:
    1. Forward through both branches to capture bottleneck features za and zb
    2. Haar transform: ca, cb = Haar(za), Haar(zb)
    3. Swap 7 HF subbands:
       za_wfe = IDWT(LL_a, HF_b)
       zb_wfe = IDWT(LL_b, HF_a)
    4. Decode WFE-replaced bottlenecks:
       ya_wfe = Branch A decoder(za_wfe)
       yb_wfe = Branch B decoder(zb_wfe)
    """
    _, za = model.branch_a(x)
    _, zb = model.branch_b(x)

    ca, sa = haar3d(za)
    cb, sb = haar3d(zb)

    # za_wfe: LL_a + HF_b
    za_wfe = inverse_haar3d((ca[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]), sa)
    # zb_wfe: LL_b + HF_a
    zb_wfe = inverse_haar3d((cb[0], ca[1], ca[2], ca[3], ca[4], ca[5], ca[6], ca[7]), sb)

    ya_wfe, _ = model.branch_a(x, replacement=za_wfe)
    yb_wfe, _ = model.branch_b(x, replacement=zb_wfe)

    return ya_wfe, yb_wfe


# ============================================================
# LOSSES & METRICS
# ============================================================

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def supervised_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)
    return bce + dice


def consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, threshold: float = CONFIDENCE_THRESHOLD) -> torch.Tensor:
    pseudo = torch.sigmoid(teacher_logits).detach()
    return F.binary_cross_entropy_with_logits(
        student_logits,
        (pseudo >= threshold).float(),
    )


def extract_boundary_band_3d(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    pred_b = extract_boundary_band_3d(probs, kernel_size=3)
    target_b = extract_boundary_band_3d(target, kernel_size=3)
    dims = tuple(range(1, logits.ndim))
    inter = (pred_b * target_b).sum(dims)
    denom = pred_b.sum(dims) + target_b.sum(dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return (1.0 - dice).mean()


def binary_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    target = target.float()
    dims = tuple(range(1, logits.ndim))
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    return ((2.0 * inter + smooth) / (denom + smooth)).mean()


def validate_wfe(model: WFE_CPS_Model, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """
    Genuine WFE validation inference using wfe_forward_single_input.
    Returns: (Dice Branch A with WFE, Dice Branch B with WFE, Dice Ensemble with WFE)
    """
    model.eval()
    scores_a, scores_b, scores_ens = [], [], []

    def wfe_predictor(patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ya_wfe, yb_wfe = wfe_forward_single_input(model, patch)
        return ya_wfe, yb_wfe

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            # Direct patch forward through genuine WFE
            ya_wfe, yb_wfe = wfe_predictor(image)
            y_ens = 0.5 * (ya_wfe + yb_wfe)

            scores_a.append(binary_dice_from_logits(ya_wfe, label).item())
            scores_b.append(binary_dice_from_logits(yb_wfe, label).item())
            scores_ens.append(binary_dice_from_logits(y_ens, label).item())

    mean_a = float(sum(scores_a) / max(1, len(scores_a)))
    mean_b = float(sum(scores_b) / max(1, len(scores_b)))
    mean_ens = float(sum(scores_ens) / max(1, len(scores_ens)))
    return mean_a, mean_b, mean_ens


def save_checkpoint(
    path: Path,
    model: WFE_CPS_Model,
    opt_a: torch.optim.Optimizer,
    opt_b: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_val_dice: float,
):
    torch.save(
        {
            "stage": 4,
            "experiment": "stage4_wfe_boundary",
            "epoch": epoch,
            "model_a_state": model.branch_a.state_dict(),
            "model_b_state": model.branch_b.state_dict(),
            "optimizer_a_state": opt_a.state_dict(),
            "optimizer_b_state": opt_b.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "config": {
                "seed": SEED,
                "patch_size": list(PATCH_SIZE),
                "foreground_probability": FOREGROUND_PROBABILITY,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "cps_weight": CPS_WEIGHT,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "lambda_boundary": LAMBDA_BOUNDARY,
                "wfe": True,
                "validation_inference": "genuine_wfe_haar_exchange",
            },
        },
        path,
    )
