"""
WaveSeg-SSL: Phase B Training Script (Planned Next Steps N2-N5)
===============================================================
Single entry point for the Phase B ablations. Every run starts from the Stage 4
CPS + WFE + Boundary protocol (WaveSeg_SSL_Stage4_Boundary.ipynb, Step 3) and
changes exactly one factor, so each result is a controlled comparison against
the Stage 4 boundary run (lambda_bnd = 0.1, val-selected, Dice 0.7533 / HD95 33.06).

  N2  Boundary loss weight tuning   --lambda-bnd {0.05, 0.2, 0.5}
  N3  nnU-Net style augmentation    --aug nnunet
  N4  Uncertainty-aware CPS         --cps-mode entropy
        w(x_u) = 1 - H(p_teacher) / log 2   (soft per-voxel weight, replaces tau = 0.7)
  N5  Foundation model third branch --third-branch swin [--peft lora|adapter]
        3D Swin-UNETR (SSL-pretrained encoder, frozen) trained as a third CPS branch
        alongside the WFE pair, fine-tuned with either
          lora    - low-rank updates on attention qkv/proj (Hu et al., 2022)
          adapter - Houlsby bottleneck adapters after attention and MLP (Houlsby et al., 2019),
                    bottleneck 12 = same trainable-parameter budget as LoRA rank 8
  Boundary loss variants             --bnd-loss morph|dist|uncert
        morph  - soft-morphological boundary Dice (Stage 4 baseline)
        dist   - distance-weighted error, sum d|p-y| / sum d (Kervadec et al., 2019)
        uncert - uncertainty-rectified boundary loss: inside a soft band around the GT
                 surface, exp(-U) * BCE + U with U = KL(branch consensus || branch)
                 (Zheng & Yang, 2021); outside the band, the distance-weighted error.
                 Ambiguous (hazy) boundary voxels, where the branches disagree, are
                 attenuated instead of being forced onto a 1-voxel annotated shell.

Model selection uses the validation split only. The locked test split is read
only by --evaluate-test, which refuses to run without --confirm-frozen.

Colab usage:
  python train_phase_b.py --exp-name n2_bnd_0p05 --lambda-bnd 0.05
  python train_phase_b.py --exp-name n3_aug      --aug nnunet
  python train_phase_b.py --exp-name n4_entropy  --cps-mode entropy
  python train_phase_b.py --exp-name n5_lora     --third-branch swin --peft lora
  python train_phase_b.py --exp-name n5_adapter  --third-branch swin --peft adapter
  python train_phase_b.py --exp-name b_uncert    --bnd-loss uncert
  python train_phase_b.py --exp-name n3_aug --resume              # after a disconnect
  python train_phase_b.py --exp-name n3_aug --evaluate-test --confirm-frozen

Local CPU smoke test (synthetic volumes, no Drive / dataset needed):
  python train_phase_b.py --smoke [--aug nnunet --cps-mode entropy --bnd-loss uncert --third-branch swin]
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

import monai
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR, UNet

monai.data.set_track_meta(False)  # transforms return plain tensors, not MetaTensors


# ============================================================
# PATHS & FIXED PROTOCOL (parity with Stage 4 boundary run)
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive")
CHECKPOINTS_ROOT = DRIVE_ROOT / "waveseg_ssl_checkpoints"
DATA_ROOT = DRIVE_ROOT / "WaveSeg-SSL_Datasets/UA-MT/data/2018LA_Seg_Training Set"
LOCAL_DATA_ROOT = Path("/content/UA-MT/data/2018LA_Seg_Training Set")
SPLIT_PATH = CHECKPOINTS_ROOT / "stage3_cps/stage3_cps_split.json"
PHASE_B_ROOT = CHECKPOINTS_ROOT / "phase_b"
SWIN_WEIGHTS_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
)
SWIN_WEIGHTS_PATH = CHECKPOINTS_ROOT / "pretrained/ssl_pretrained_weights.pth"

PATCH_SIZE = (96, 96, 80)
CPS_WEIGHT = 1.0
WEIGHT_DECAY = 1e-5


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WaveSeg-SSL Phase B training (N2-N5)")
    p.add_argument("--exp-name", type=str, default=None, help="Output folder under phase_b/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    # N2
    p.add_argument("--lambda-bnd", type=float, default=0.1, help="Boundary loss weight")
    p.add_argument("--bnd-loss", choices=["morph", "dist", "uncert"], default="morph")
    p.add_argument("--bnd-band", type=float, default=2.0,
                   help="uncert: Gaussian half-width (voxels) of the near-surface band")
    # N3
    p.add_argument("--aug", choices=["none", "nnunet"], default="none")
    p.add_argument("--crop", choices=["center", "fg_random"], default="fg_random",
                   help="'fg_random' = foreground-biased random crop (train_wfe_boundary.py). "
                        "'center' (fixed crop from the Stage 4 notebook) overfits: val Dice peaks ~0.39 at epoch 5")
    # N4
    p.add_argument("--cps-mode", choices=["threshold", "entropy"], default="threshold")
    p.add_argument("--cps-threshold", type=float, default=0.7, help="Confidence tau for --cps-mode threshold")
    # N5
    p.add_argument("--third-branch", choices=["none", "swin", "swin_lora"], default="none",
                   help="'swin_lora' is kept as an alias for '--third-branch swin --peft lora'")
    p.add_argument("--peft", choices=["lora", "adapter"], default="lora")
    p.add_argument("--swin-weights", type=str, default=str(SWIN_WEIGHTS_PATH))
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--adapter-dim", type=int, default=12, help="Adapter bottleneck width")
    # Run control
    p.add_argument("--resume", action="store_true", help="Continue from latest.pt in the experiment folder")
    p.add_argument("--evaluate-test", action="store_true", help="Evaluate best.pt on the locked test split")
    p.add_argument("--confirm-frozen", action="store_true", help="Required with --evaluate-test")
    p.add_argument("--smoke", action="store_true", help="CPU smoke test on synthetic volumes")
    p.add_argument("--allow-cpu", action="store_true", help="Train even without a GPU (~20x slower)")
    args = p.parse_args(argv)
    if not args.smoke and not args.exp_name:
        p.error("--exp-name is required (except with --smoke)")
    if args.third_branch == "swin_lora":
        if args.peft != "lora":
            p.error("--third-branch swin_lora implies --peft lora; use --third-branch swin")
        args.third_branch = "swin"
    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# DATA
# ============================================================

def resolve_h5(entry: dict) -> Path:
    p = Path(entry.get("h5_path", ""))
    if p.exists():
        return p
    for root in (DATA_ROOT, LOCAL_DATA_ROOT):
        cand = root / entry["case_id"] / "mri_norm2.h5"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not locate mri_norm2.h5 for case {entry.get('case_id')}")


def synthetic_volume(index: int, shape=(104, 100, 88)) -> Tuple[np.ndarray, np.ndarray]:
    """Noisy ellipsoid used by --smoke; odd-ish shape exercises padding and sliding windows."""
    rng = np.random.default_rng(index)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    c = np.array(shape) / 2 + rng.integers(-8, 8, size=3)
    r = np.array([28, 24, 20])
    label = ((((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2) <= 1).astype(np.uint8)
    image = (label * 1.5 + rng.normal(0, 0.5, shape)).astype(np.float32)
    return image, label


def load_volume(entry: dict) -> Tuple[np.ndarray, np.ndarray]:
    if "synthetic" in entry:
        return synthetic_volume(entry["synthetic"])
    with h5py.File(resolve_h5(entry), "r") as f:
        return np.asarray(f["image"], dtype=np.float32), np.asarray(f["label"], dtype=np.uint8)


def center_crop(image: np.ndarray, label: np.ndarray, patch: Tuple[int, int, int]):
    """Exact crop used by PatchDataset in the Stage 4 boundary notebook."""
    starts = [max(0, (s - t) // 2) for s, t in zip(image.shape, patch)]
    sl = tuple(slice(st, st + t) for st, t in zip(starts, patch))
    image, label = image[sl], label[sl]
    pads = [(max(0, t - d) // 2, max(0, t - d) - max(0, t - d) // 2) for d, t in zip(image.shape, patch)]
    if any(a or b for a, b in pads):
        image = np.pad(image, pads, mode="reflect")
        label = np.pad(label, pads, mode="constant", constant_values=0)
    return image, label


def fg_random_crop(image: np.ndarray, label: np.ndarray, patch: Tuple[int, int, int],
                   rng: np.random.Generator, fg_prob: float = 0.75):
    """Foreground-biased random crop (train_wfe_boundary.py helper)."""
    pads = [(max(0, t - c) // 2, max(0, t - c) - max(0, t - c) // 2) for c, t in zip(image.shape, patch)]
    image, label = np.pad(image, pads), np.pad(label, pads)
    fg = np.argwhere(label > 0)
    center = (fg[rng.integers(len(fg))] if len(fg) and rng.random() < fg_prob
              else np.array([rng.integers(n) for n in image.shape]))
    starts = [int(np.clip(center[i] - patch[i] // 2, 0, image.shape[i] - patch[i])) for i in range(3)]
    sl = tuple(slice(s, s + t) for s, t in zip(starts, patch))
    return image[sl], label[sl]


def build_nnunet_augmentation(with_label: bool):
    """
    N3: nnU-Net style spatial + intensity augmentation (Isensee et al., 2021),
    applied to the 96x96x80 training patch. Labels use nearest-neighbour
    interpolation; intensity transforms touch the image only.
    """
    from monai.transforms import (
        Compose, Rand3DElasticd, RandAdjustContrastd, RandFlipd, RandGaussianNoised,
        RandGaussianSmoothd, RandRotated, RandScaleIntensityd, RandSimulateLowResolutiond,
        RandZoomd,
    )
    keys = ["image", "label"] if with_label else ["image"]
    interp = ("bilinear", "nearest") if with_label else "bilinear"
    zoom_interp = ("trilinear", "nearest") if with_label else "trilinear"
    rot = math.pi / 6  # +/- 30 degrees
    return Compose([
        RandRotated(keys, range_x=rot, range_y=rot, range_z=rot, prob=0.2,
                    mode=interp, padding_mode="reflection"),
        RandZoomd(keys, min_zoom=0.7, max_zoom=1.4, prob=0.2, mode=zoom_interp, keep_size=True),
        Rand3DElasticd(keys, sigma_range=(5, 8), magnitude_range=(100, 200), prob=0.1,
                       mode=interp, padding_mode="reflection"),
        RandGaussianNoised(["image"], prob=0.1, mean=0.0, std=0.1),
        RandGaussianSmoothd(["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        RandScaleIntensityd(["image"], factors=0.25, prob=0.15),
        RandSimulateLowResolutiond(["image"], prob=0.15, zoom_range=(0.5, 1.0)),
        RandAdjustContrastd(["image"], prob=0.3, gamma=(0.7, 1.5)),
        RandFlipd(keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys, spatial_axis=1, prob=0.5),
        RandFlipd(keys, spatial_axis=2, prob=0.5),
    ])


def surface_distance_map(label: np.ndarray, empty_value: float = 50.0) -> np.ndarray:
    """
    Unsigned Euclidean distance (voxels) from every voxel to the GT surface, 0.5 on the
    voxels either side of the interface. Patches without a surface get a constant
    large distance, so every foreground prediction there counts as a far error.
    """
    from scipy.ndimage import distance_transform_edt

    fg = label > 0.5
    if not fg.any() or fg.all():
        return np.full(label.shape, empty_value, dtype=np.float32)
    return (distance_transform_edt(~fg) + distance_transform_edt(fg) - 0.5).astype(np.float32)


class PatchDataset(Dataset):
    def __init__(self, entries: List[dict], *, labeled: bool, crop: str, aug: str, seed: int,
                 with_dist: bool = False):
        self.entries = entries
        self.labeled = labeled
        self.with_dist = labeled and with_dist
        self.crop = crop
        self.rng = np.random.default_rng(seed)
        self.augment = build_nnunet_augmentation(labeled) if aug == "nnunet" else None
        if self.augment is not None:
            self.augment.set_random_state(seed)

    def reseed(self, seed: int):
        self.rng = np.random.default_rng(seed)
        if self.augment is not None:
            self.augment.set_random_state(seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image, label = load_volume(self.entries[idx])
        if self.crop == "center":
            image, label = center_crop(image, label, PATCH_SIZE)
        else:
            image, label = fg_random_crop(image, label, PATCH_SIZE, self.rng)
        sample = {"image": torch.from_numpy(np.ascontiguousarray(image[None])).float()}
        if self.labeled:
            sample["label"] = torch.from_numpy(np.ascontiguousarray(label[None])).float()
        if self.augment is not None:
            sample = self.augment(sample)
            sample["image"] = torch.as_tensor(sample["image"]).float()
            if self.labeled:
                sample["label"] = (torch.as_tensor(sample["label"]) > 0.5).float()
        if self.with_dist:
            sample["dist"] = torch.from_numpy(surface_distance_map(sample["label"][0].numpy())[None])
        return sample


class VolumeDataset(Dataset):
    def __init__(self, entries: List[dict]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image, label = load_volume(self.entries[idx])
        return {"image": torch.from_numpy(image[None]).float(), "label": torch.from_numpy(label[None]).float()}


def worker_init_fn(worker_id: int):
    """Give each worker its own crop/augmentation stream (otherwise forked workers repeat draws)."""
    info = get_worker_info()
    seed = int(torch.initial_seed() % (2 ** 32))
    np.random.seed(seed)
    info.dataset.reseed(seed)


# ============================================================
# 3D HAAR WAVELET FREQUENCY EXCHANGE (unchanged from Stage 4)
# ============================================================

def haar3d(x: torch.Tensor):
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


def inverse_haar3d(coeffs, orig):
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


def wavelet_frequency_exchange(za, zb):
    ca, sa = haar3d(za)
    cb, sb = haar3d(zb)
    za_wfe = inverse_haar3d((ca[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7]), sa)
    zb_wfe = inverse_haar3d((cb[0], ca[1], ca[2], ca[3], ca[4], ca[5], ca[6], ca[7]), sb)
    return za_wfe, zb_wfe


class WFEBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(spatial_dims=3, in_channels=1, out_channels=1,
                         channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)

    def get_bottleneck(self):
        return self.unet.model[1].submodule[1].submodule[1].submodule[1].submodule

    def forward(self, x, replacement=None):
        captured = {}

        def hook(m, inp, outp):
            captured["z"] = outp
            return replacement if replacement is not None else outp

        handle = self.get_bottleneck().register_forward_hook(hook)
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

    def forward(self, xa, xb):
        _, za = self.branch_a(xa)
        _, zb = self.branch_b(xb)
        za_wfe, zb_wfe = wavelet_frequency_exchange(za, zb)
        ya, _ = self.branch_a(xa, replacement=za_wfe)
        yb, _ = self.branch_b(xb, replacement=zb_wfe)
        return ya, yb


# ============================================================
# N5: SWIN-UNETR + LoRA THIRD BRANCH
# ============================================================

class LoRALinear(nn.Module):
    """
    W x + (alpha / r) * B A x, wrapping an existing pretrained nn.Linear.
    W (and bias) stay frozen; A is Kaiming-initialised and B is zero, so the
    adapted layer starts out identical to the pretrained one.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def inject_lora(module: nn.Module, r: int, alpha: float, targets=("qkv", "proj")) -> int:
    count = 0
    for parent in list(module.modules()):
        for name in targets:
            child = getattr(parent, name, None)
            if isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha))
                count += 1
    return count


class Adapter(nn.Module):
    """
    Houlsby bottleneck adapter on a frozen sublayer's output: h + up(GELU(down(h))).
    `up` is zero-initialised, so the adapted block starts identical to the pretrained one.
    Unlike LoRA, the extra layers stay in the forward pass at inference (cannot be merged).
    """

    def __init__(self, sublayer: nn.Module, dim: int, bottleneck: int):
        super().__init__()
        self.sublayer = sublayer
        for p in self.sublayer.parameters():
            p.requires_grad_(False)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = self.sublayer(x)
        return h + self.up(F.gelu(self.down(h)))


def inject_adapters(swin: nn.Module, bottleneck: int) -> int:
    """Adapters after the attention output projection and after the MLP of every Swin block."""
    count = 0
    for blk in list(swin.modules()):
        if hasattr(blk, "attn") and hasattr(blk, "mlp") and hasattr(blk, "dim"):
            blk.attn.proj = Adapter(blk.attn.proj, blk.dim, bottleneck)
            blk.mlp = Adapter(blk.mlp, blk.dim, bottleneck)
            count += 2
    return count


def load_swin_ssl_weights(net: SwinUNETR, path: Path) -> Tuple[int, int]:
    """Load MONAI's self-supervised Swin-UNETR encoder weights into net.swinViT."""
    from monai.networks.nets.swin_unetr import filter_swinunetr
    from monai.networks.utils import copy_model_state

    if not path.exists():
        from monai.apps import download_url
        path.parent.mkdir(parents=True, exist_ok=True)
        download_url(SWIN_WEIGHTS_URL, str(path))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ssl = ckpt.get("model", ckpt.get("state_dict", ckpt))
    _, loaded, _ = copy_model_state(net, ssl, filter_func=filter_swinunetr)
    n_vit = sum(1 for k in net.state_dict() if k.startswith("swinViT."))
    n_loaded = sum(1 for k in loaded if k.startswith("swinViT."))
    if n_loaded < n_vit:
        raise RuntimeError(f"Swin encoder only partially initialised: {n_loaded}/{n_vit} tensors")
    return n_loaded, n_vit


class SwinPEFTBranch(nn.Module):
    """
    Third CPS branch. The pretrained Swin transformer encoder is frozen and adapted
    through LoRA or bottleneck adapters; the convolutional skip encoders and decoder
    (randomly initialised in the SSL checkpoint) are fully trainable. Inputs are
    reflect-padded to a multiple of 32 (80 -> 96 on the z-axis) and the logits are
    cropped back.
    """

    def __init__(self, weights_path: Optional[Path], peft: str, r: int, alpha: float, adapter_dim: int,
                 use_checkpoint: bool = True):
        super().__init__()
        self.net = SwinUNETR(in_channels=1, out_channels=1, feature_size=48, use_checkpoint=use_checkpoint)
        self.pretrained = None
        if weights_path is not None:
            self.pretrained = load_swin_ssl_weights(self.net, weights_path)
        for p in self.net.swinViT.parameters():
            p.requires_grad_(False)
        self.peft = peft
        self.encoder_params_before = sum(p.numel() for p in self.net.swinViT.parameters())
        if peft == "lora":
            self.n_peft = inject_lora(self.net.swinViT, r, alpha)
        else:
            self.n_peft = inject_adapters(self.net.swinViT, adapter_dim)
        self.peft_params = sum(p.numel() for p in self.net.swinViT.parameters()) - self.encoder_params_before

    def forward(self, x):
        shape = x.shape[-3:]
        pads = [(-s) % 32 for s in shape]
        if any(pads):
            x = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]), mode="reflect")
        return self.net(x)[..., :shape[0], :shape[1], :shape[2]]


# ============================================================
# LOSSES
# ============================================================

def dice_loss(logits, target, smooth=1e-5):
    p = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    inter = (p * target).sum(dims)
    card = p.sum(dims) + target.sum(dims)
    return 1.0 - ((2.0 * inter + smooth) / (card + smooth)).mean()


def extract_boundary_band_3d(mask, kernel_size=3):
    pad = kernel_size // 2
    dilated = F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_dice_loss(logits, target, smooth=1e-5):
    pred_b = extract_boundary_band_3d(torch.sigmoid(logits))
    target_b = extract_boundary_band_3d(target.float())
    dims = tuple(range(1, logits.ndim))
    inter = (pred_b * target_b).sum(dims)
    card = pred_b.sum(dims) + target_b.sum(dims)
    return (1.0 - (2.0 * inter + smooth) / (card + smooth)).mean()


def distance_boundary_loss(logits, target, dist, eps=1e-6):
    """
    Boundary loss of Kervadec et al. (2019), written in its non-negative form
    sum d|p - y| / sum d, where d is the distance to the GT surface. Errors on the
    surface itself cost ~nothing (tolerant of 1-voxel annotation ambiguity); errors
    k voxels away cost k times more (directly targets HD95). No morphology, and
    every voxel receives a gradient.
    """
    p = torch.sigmoid(logits.float())
    dims = tuple(range(1, logits.ndim))
    return ((dist * (p - target).abs()).sum(dims) / (dist.sum(dims) + eps)).mean()


def binary_kl(p, q, eps=1e-6):
    p, q = p.clamp(eps, 1.0 - eps), q.clamp(eps, 1.0 - eps)
    return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))


def uncertainty_boundary_loss(logits, target, dist, consensus, band: float = 2.0, eps=1e-6):
    """
    Uncertainty-rectified boundary loss for hazy MRI/CT boundaries.

      g    = exp(-d^2 / 2 band^2)       soft band around the GT surface (from GT distances only)
      U    = KL(consensus || p)          disagreement of this branch with the detached
                                         consensus of all branches = boundary ambiguity
      near = sum g [exp(-U) BCE(p, y) + U] / sum g      (Zheng & Yang, IJCV 2021)
      far  = sum (1-g) d |p - y| / sum (1-g) d          (Kervadec et al., 2019)

    Near the surface, voxels where the branches disagree have their BCE attenuated
    instead of being forced onto the annotated shell; the +U term stops the network
    from buying attenuation by staying uncertain. Far from the surface nothing is
    attenuated - distant errors are penalised in proportion to their distance.
    Returns (loss, mean U inside the band).
    """
    logits = logits.float()
    p = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    g = torch.exp(-dist.pow(2) / (2.0 * band ** 2))
    u = binary_kl(consensus.detach(), p)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    near = (g * (torch.exp(-u) * bce + u)).sum(dims) / (g.sum(dims) + eps)
    far_w = (1.0 - g) * dist
    far = (far_w * (p - target).abs()).sum(dims) / (far_w.sum(dims) + eps)
    u_band = ((g * u).sum(dims) / (g.sum(dims) + eps)).mean()
    return (near + far).mean(), u_band


def binary_entropy(p, eps=1e-6):
    """H(p) / log 2, i.e. normalised to [0, 1]."""
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p))


def cps_term(student_logits, teacher_prob, mode: str, threshold: float):
    """
    Pseudo-label BCE from a (detached) teacher probability map.
      threshold: w = 1[max(p, 1-p) >= tau]            (Stage 3/4 baseline)
      entropy:   w = 1 - H(p) / log 2                  (N4, soft per-voxel weight)
    Returns (loss, mean weight) - the mean weight is the effective pseudo-label coverage.
    """
    teacher_prob = teacher_prob.detach().float()
    pseudo = (teacher_prob >= 0.5).float()
    if mode == "threshold":
        w = (torch.maximum(teacher_prob, 1.0 - teacher_prob) >= threshold).float()
    else:
        w = 1.0 - binary_entropy(teacher_prob)
    bce = F.binary_cross_entropy_with_logits(student_logits.float(), pseudo, reduction="none")
    return (bce * w).sum() / (w.sum() + 1e-6), w.mean()


# ============================================================
# MODEL WRAPPER
# ============================================================

class PhaseBModel(nn.Module):
    def __init__(self, args, weights_path: Optional[Path]):
        super().__init__()
        self.wfe = WFE_CPS_Model()
        self.branch_c = (SwinPEFTBranch(weights_path, args.peft, args.lora_rank, args.lora_alpha, args.adapter_dim)
                         if args.third_branch == "swin" else None)

    def forward(self, x) -> List[torch.Tensor]:
        ya, yb = self.wfe(x, x)
        return [ya, yb] if self.branch_c is None else [ya, yb, self.branch_c(x)]

    def predictors(self):
        """Per-branch whole-volume predictors (plain UNets for A/B, as in Stage 4 validation)."""
        preds = {"a": self.wfe.branch_a.unet, "b": self.wfe.branch_b.unet}
        if self.branch_c is not None:
            preds["c"] = self.branch_c
        return preds


def make_optimizers(model: PhaseBModel, lr: float):
    groups = {"a": model.wfe.branch_a.parameters(), "b": model.wfe.branch_b.parameters()}
    if model.branch_c is not None:
        groups["c"] = [p for p in model.branch_c.parameters() if p.requires_grad]
    return {k: torch.optim.AdamW(v, lr=lr, weight_decay=WEIGHT_DECAY) for k, v in groups.items()}


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def predict_probs(model: PhaseBModel, image, dev) -> Dict[str, torch.Tensor]:
    out = {}
    for name, fn in model.predictors().items():
        logits = sliding_window_inference(image, PATCH_SIZE, 1, fn, overlap=0.25)
        out[name] = torch.sigmoid(logits.float())
    return out


def dice_of(pred, label, smooth=1e-5):
    dims = tuple(range(1, label.ndim))
    inter = (pred * label).sum(dims)
    card = pred.sum(dims) + label.sum(dims)
    return float(((2.0 * inter + smooth) / (card + smooth)).mean().item())


def ensembles(probs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {"ab": 0.5 * (probs["a"] + probs["b"])}
    if "c" in probs:
        out["abc"] = (probs["a"] + probs["b"] + probs["c"]) / 3.0
        out["c"] = probs["c"]
    return out


@torch.no_grad()
def run_validation(model: PhaseBModel, loader, dev) -> Dict[str, float]:
    model.eval()
    scores: Dict[str, List[float]] = {}
    for b in loader:
        im, lb = b["image"].to(dev), b["label"].to(dev)
        for name, prob in ensembles(predict_probs(model, im, dev)).items():
            scores.setdefault(name, []).append(dice_of((prob >= 0.5).float(), lb))
    res = {k: float(np.mean(v)) for k, v in scores.items()}
    # Selection metric: ensemble of every branch that is trained
    res["select"] = res.get("abc", res["ab"])
    return res


def evaluate_test(model: PhaseBModel, entries: List[dict], dev, out_path: Path, ckpt: dict):
    from monai.metrics import compute_hausdorff_distance

    loader = DataLoader(VolumeDataset(entries), batch_size=1, shuffle=False)
    per_variant: Dict[str, List[dict]] = {}
    model.eval()
    with torch.no_grad():
        for idx, b in enumerate(loader):
            cid = entries[idx].get("case_id", str(idx))
            im, lb = b["image"].to(dev), b["label"].to(dev)
            for name, prob in ensembles(predict_probs(model, im, dev)).items():
                pred = (prob >= 0.5).float()
                p_np, t_np = pred[0, 0].cpu().numpy().astype(bool), lb[0, 0].cpu().numpy().astype(bool)
                tp = float(np.logical_and(p_np, t_np).sum())
                fp = float(np.logical_and(p_np, ~t_np).sum())
                fn = float(np.logical_and(~p_np, t_np).sum())
                try:
                    hd = float(compute_hausdorff_distance(pred, lb, include_background=False, percentile=95.0)[0, 0])
                    hd = hd if np.isfinite(hd) else float("nan")
                except Exception:
                    hd = float("nan")
                per_variant.setdefault(name, []).append({
                    "case_id": cid, "dice": dice_of(pred, lb),
                    "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
                    "recall": tp / (tp + fn) if tp + fn > 0 else 0.0,
                    "hd95": hd,
                })
            d = per_variant["ab"][-1]
            print(f"[{idx + 1:02d}/{len(entries)}] {cid:20s} | AB Dice {d['dice']:.4f} | HD95 {d['hd95']:.2f}")

    summary = {"best_epoch": ckpt.get("best_epoch"), "best_val": ckpt.get("best_val"),
               "config": ckpt.get("config"), "variants": {}}
    for name, cases in per_variant.items():
        hd = [c["hd95"] for c in cases if np.isfinite(c["hd95"])]
        summary["variants"][name] = {
            "mean_dice": float(np.mean([c["dice"] for c in cases])),
            "std_dice": float(np.std([c["dice"] for c in cases])),
            "mean_precision": float(np.mean([c["precision"] for c in cases])),
            "mean_recall": float(np.mean([c["recall"] for c in cases])),
            "mean_hd95": float(np.mean(hd)) if hd else None,
            "per_case": cases,
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print(f"{'Predictor':<12} | {'Dice':<8} | {'Prec':<8} | {'Recall':<8} | {'HD95':<8}")
    print("-" * 75)
    for name, s in summary["variants"].items():
        hd = f"{s['mean_hd95']:.3f}" if s["mean_hd95"] is not None else "N/A"
        print(f"{name.upper():<12} | {s['mean_dice']:<8.4f} | {s['mean_precision']:<8.4f} | {s['mean_recall']:<8.4f} | {hd:<8}")
    print("=" * 75)
    print(f"Saved: {out_path}")


# ============================================================
# CHECKPOINTING
# ============================================================

def model_state(model: PhaseBModel) -> dict:
    state = {"model_a_state": model.wfe.branch_a.state_dict(), "model_b_state": model.wfe.branch_b.state_dict()}
    if model.branch_c is not None:
        # Frozen pretrained weights are re-loadable, but storing them keeps the checkpoint self-contained
        state["model_c_state"] = model.branch_c.state_dict()
    return state


def load_model_state(model: PhaseBModel, ckpt: dict):
    model.wfe.branch_a.load_state_dict(ckpt["model_a_state"])
    model.wfe.branch_b.load_state_dict(ckpt["model_b_state"])
    if model.branch_c is not None:
        model.branch_c.load_state_dict(ckpt["model_c_state"])


# Keys added after the first Phase B runs; older checkpoints are treated as having these values.
LEGACY_CONFIG_DEFAULTS = {"bnd_loss": "morph", "bnd_band": 2.0, "peft": "lora", "adapter_dim": 12}


def run_config(args) -> dict:
    return {
        "phase": "B", "exp_name": args.exp_name, "epochs": args.epochs, "learning_rate": args.lr,
        "seed": args.seed, "patch_size": list(PATCH_SIZE), "crop": args.crop, "aug": args.aug,
        "cps_weight": CPS_WEIGHT, "cps_mode": args.cps_mode, "cps_threshold": args.cps_threshold,
        "lambda_boundary": args.lambda_bnd, "bnd_loss": args.bnd_loss, "bnd_band": args.bnd_band,
        "third_branch": args.third_branch, "peft": args.peft,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "adapter_dim": args.adapter_dim,
        "wfe": True, "init_mode": "from_scratch",
        "validation_inference": "sliding_window_plain_unet_ensemble",
        "selection": "val ensemble Dice (ABC if third branch else AB)",
        "test_used_for_tuning": False,
    }


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"

    if args.smoke:
        args.exp_name = args.exp_name or "smoke"
        args.epochs = min(args.epochs, 2)
        args.num_workers = 0
        exp_dir = Path(__file__).resolve().parent / "_smoke_runs" / args.exp_name
        splits = {
            "labeled": [{"case_id": f"syn{i}", "synthetic": i} for i in range(2)],
            "unlabeled": [{"case_id": f"syn{i}", "synthetic": i} for i in range(2, 5)],
            "val": [{"case_id": "syn5", "synthetic": 5}],
            "test": [{"case_id": "syn6", "synthetic": 6}],
        }
        weights_path = Path(args.swin_weights) if Path(args.swin_weights).exists() else None
    else:
        exp_dir = PHASE_B_ROOT / args.exp_name
        with open(SPLIT_PATH) as f:
            splits = json.load(f)["splits"]
        assert (len(splits["labeled"]), len(splits["unlabeled"]), len(splits["val"]), len(splits["test"])) == (14, 56, 15, 15), \
            "Locked split sizes changed - refusing to run"
        weights_path = Path(args.swin_weights)
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = PhaseBModel(args, weights_path).to(dev)

    print("=" * 80)
    print(f"WAVESEG-SSL PHASE B | {args.exp_name} | device={dev} | amp={use_amp}")
    print(f"bnd_loss={args.bnd_loss} (lambda={args.lambda_bnd}) | aug={args.aug} | crop={args.crop} | cps={args.cps_mode}"
          f"{f' (tau={args.cps_threshold})' if args.cps_mode == 'threshold' else ''} | third={args.third_branch}")
    if model.branch_c is not None:
        c = model.branch_c
        trainable = sum(p.numel() for p in c.parameters() if p.requires_grad)
        total = sum(p.numel() for p in c.parameters())
        pre = f"{c.pretrained[0]}/{c.pretrained[1]} encoder tensors loaded" if c.pretrained else "RANDOM INIT (smoke only)"
        print(f"Swin-UNETR branch: {pre} | {c.peft} modules={c.n_peft} ({c.peft_params / 1e3:.0f}K params) | "
              f"trainable {trainable / 1e6:.2f}M / {total / 1e6:.2f}M (incl. decoder)")
    print(f"Output: {exp_dir}")
    print("=" * 80)

    # ---------------- Locked test evaluation ----------------
    if args.evaluate_test:
        if not args.confirm_frozen:
            raise RuntimeError("Safety guard: pass --confirm-frozen only after the configuration is final.")
        ckpt = torch.load(exp_dir / "best.pt", map_location=dev, weights_only=False)
        load_model_state(model, ckpt)
        evaluate_test(model, splits["test"], dev, exp_dir / "test_metrics_detailed.json", ckpt)
        return

    # ---------------- Training ----------------
    if dev.type == "cpu" and not (args.smoke or args.allow_cpu):
        raise SystemExit("No GPU detected - training on CPU takes ~13 min/epoch (~22 h per run). "
                         "In Colab: Runtime > Change runtime type > GPU, then rerun with --resume. "
                         "Pass --allow-cpu to train on CPU anyway.")
    loader_kw = dict(batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=dev.type == "cuda",
                     worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
                     persistent_workers=args.num_workers > 0)
    train_l = DataLoader(PatchDataset(splits["labeled"], labeled=True, crop=args.crop, aug=args.aug, seed=args.seed,
                                      with_dist=args.bnd_loss != "morph"), **loader_kw)
    train_u = DataLoader(PatchDataset(splits["unlabeled"], labeled=False, crop=args.crop, aug=args.aug, seed=args.seed + 1), **loader_kw)
    val_loader = DataLoader(VolumeDataset(splits["val"]), batch_size=1, shuffle=False)

    opts = make_optimizers(model, args.lr)
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)

    history, best_val, best_epoch, start_epoch = [], -1.0, 0, 1
    if args.resume and (exp_dir / "latest.pt").exists():
        ck = torch.load(exp_dir / "latest.pt", map_location=dev, weights_only=False)
        changed = {k for k, v in run_config(args).items()
                   if k != "epochs" and ck["config"].get(k, LEGACY_CONFIG_DEFAULTS.get(k)) != v}
        if changed:
            raise RuntimeError(f"latest.pt was written with a different configuration ({sorted(changed)}) - refusing to resume")
        load_model_state(model, ck)
        for k, o in opts.items():
            o.load_state_dict(ck["optimizer_states"][k])
        # Empty when latest.pt was written on CPU (AMP disabled); the GPU scaler then starts fresh.
        if ck["scaler_state"] and use_amp:
            scaler.load_state_dict(ck["scaler_state"])
        history, best_val, best_epoch = ck["history"], ck["best_val"], ck["best_epoch"]
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from epoch {ck['epoch']} (best val {best_val:.6f} @ {best_epoch})")
    elif (exp_dir / "latest.pt").exists() and not args.smoke:
        raise RuntimeError(f"{exp_dir} already has a run - use --resume or a new --exp-name")

    with open(exp_dir / "config.json", "w") as f:
        json.dump(run_config(args), f, indent=2)

    steps_per_epoch = max(len(train_l), len(train_u))
    if args.smoke:
        steps_per_epoch = 2
    iter_l, iter_u = iter(train_l), iter(train_u)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        acc = {"loss": 0.0, "sup": 0.0, "cps": 0.0, "bnd": 0.0, "cps_w": 0.0, "bnd_u": 0.0}
        for _ in range(steps_per_epoch):
            try:
                bl = next(iter_l)
            except StopIteration:
                iter_l = iter(train_l)
                bl = next(iter_l)
            try:
                bu = next(iter_u)
            except StopIteration:
                iter_u = iter(train_u)
                bu = next(iter_u)
            xl, yl = bl["image"].to(dev, non_blocking=True), bl["label"].to(dev, non_blocking=True)
            xu = bu["image"].to(dev, non_blocking=True)

            for o in opts.values():
                o.zero_grad(set_to_none=True)

            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits_l = model(xl)
                logits_u = model(xu)
                n = len(logits_l)

                sup = sum(F.binary_cross_entropy_with_logits(lg, yl) + dice_loss(lg, yl) for lg in logits_l) / n
                bnd_u = torch.zeros((), device=dev)
                if args.bnd_loss == "morph":
                    bnd = sum(boundary_dice_loss(lg, yl) for lg in logits_l) / n
                elif args.bnd_loss == "dist":
                    dl = bl["dist"].to(dev, non_blocking=True)
                    bnd = sum(distance_boundary_loss(lg, yl, dl) for lg in logits_l) / n
                else:
                    dl = bl["dist"].to(dev, non_blocking=True)
                    consensus = torch.stack([torch.sigmoid(lg.float()) for lg in logits_l]).mean(0).detach()
                    bnd = 0.0
                    for lg in logits_l:
                        term, u = uncertainty_boundary_loss(lg, yl, dl, consensus, args.bnd_band)
                        bnd, bnd_u = bnd + term / n, bnd_u + u.detach() / n

                # Each branch learns from the mean probability of the other branches.
                # With two branches this is exactly the Stage 4 A<->B CPS.
                probs_u = [torch.sigmoid(lg.float()).detach() for lg in logits_u]
                cps, cps_w = 0.0, 0.0
                for i, lg in enumerate(logits_u):
                    teacher = sum(p for j, p in enumerate(probs_u) if j != i) / (n - 1)
                    term, w = cps_term(lg, teacher, args.cps_mode, args.cps_threshold)
                    cps, cps_w = cps + term / n, cps_w + w / n

                loss = sup + CPS_WEIGHT * cps + args.lambda_bnd * bnd

            scaler.scale(loss).backward()
            for o in opts.values():
                scaler.step(o)
            scaler.update()

            acc["loss"] += loss.item()
            acc["sup"] += sup.item()
            acc["cps"] += cps.item()
            acc["bnd"] += bnd.item()
            acc["cps_w"] += cps_w.item()
            acc["bnd_u"] += bnd_u.item()

        val = run_validation(model, val_loader, dev)
        is_best = val["select"] > best_val
        if is_best:
            best_val, best_epoch = val["select"], epoch

        rec = {"epoch": epoch, **{k: v / steps_per_epoch for k, v in acc.items()},
               **{f"val_dice_{k}": v for k, v in val.items() if k != "select"},
               "val_dice": val["select"], "is_best": is_best, "seconds": round(time.time() - t0, 1)}
        history.append(rec)

        state = {"epoch": epoch, "experiment": f"phase_b/{args.exp_name}", "config": run_config(args),
                 **model_state(model), "val_dice": val["select"], "best_val": best_val,
                 "best_val_dice": best_val, "best_epoch": best_epoch, "history": history}
        if is_best:
            torch.save(state, exp_dir / "best.pt")
        state.update({"optimizer_states": {k: o.state_dict() for k, o in opts.items()},
                      "scaler_state": scaler.state_dict()})
        torch.save(state, exp_dir / "latest.pt")
        with open(exp_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        extra = " ".join(f"{k.upper()}={v:.4f}" for k, v in val.items() if k != "select")
        bnd_u_txt = f", Bnd-U {rec['bnd_u']:.4f}" if args.bnd_loss == "uncert" else ""
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Loss {rec['loss']:.4f} (Sup {rec['sup']:.4f}, "
              f"CPS {rec['cps']:.4f}, Bnd {rec['bnd']:.4f}, CPS-w {rec['cps_w']:.3f}{bnd_u_txt}) | Val {extra} | "
              f"Best {best_val:.6f} (Ep {best_epoch}) | {rec['seconds']}s{' *NEW BEST*' if is_best else ''}")

    print("=" * 80)
    print(f"TRAINING COMPLETE. Best val Dice {best_val:.6f} (epoch {best_epoch}). Test set NOT touched.")
    print("=" * 80)


if __name__ == "__main__":
    main()
