"""Stage 2 preprocessing and patch sampling.

No resampling or spacing-dependent operation is performed. By default images
are unchanged because UA-MT ``mri_norm2`` is already preprocessed. Optional
per-volume z-score normalization is available for experiments.
"""
from __future__ import annotations
import numpy as np

class PreprocessingNotApprovedError(RuntimeError):
    """Backward-compatible Stage 1 exception."""

def build_preprocessing_transforms(*args: object, **kwargs: object) -> None:
    raise PreprocessingNotApprovedError("Stage 1 inspection does not use transforms.")

def normalize_image(image: np.ndarray, mode: str = "none") -> np.ndarray:
    if mode == "none": return image.astype(np.float32, copy=False)
    if mode != "zscore": raise ValueError("normalization must be 'none' or 'zscore'.")
    return ((image - image.mean()) / max(float(image.std()), 1e-8)).astype(np.float32, copy=False)

def foreground_or_random_crop(image: np.ndarray, label: np.ndarray, patch_size: tuple[int, int, int], foreground_probability: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Metadata-free fixed crop, foreground-centered with requested probability."""
    if image.shape != label.shape: raise ValueError("Image and label must have matching shapes.")
    pads = [(max(0, size - cur) // 2, max(0, size - cur) - max(0, size - cur) // 2) for cur, size in zip(image.shape, patch_size)]
    image, label = np.pad(image, pads), np.pad(label, pads)
    fg = np.argwhere(label > 0)
    center = fg[rng.integers(len(fg))] if len(fg) and rng.random() < foreground_probability else np.array([rng.integers(n) for n in image.shape])
    starts = [int(np.clip(center[i] - patch_size[i] // 2, 0, image.shape[i] - patch_size[i])) for i in range(3)]
    crop = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    return image[crop], label[crop]
