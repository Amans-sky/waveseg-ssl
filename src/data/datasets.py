"""Stage 1 manifest and lazy Stage 2 UA-MT HDF5 dataset access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import foreground_or_random_crop, normalize_image


class ManifestPairDataset(Dataset[dict[str, str]]):
    """Expose validated manifest pairs without loading or transforming volumes."""

    def __init__(self, manifest_path: str | Path) -> None:
        report: dict[str, Any] = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if report.get("inspection_only") is not True:
            raise ValueError("Expected a Stage 1 inspection manifest.")
        self._pairs: list[dict[str, str]] = report.get("proposed_pairs", [])

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, str]:
        return dict(self._pairs[index])


class UAMTH5Dataset(Dataset[dict[str, torch.Tensor | str]]):
    """Read one HDF5 volume on each access; volume data is never cached."""
    def __init__(self, entries: list[dict[str, str]], *, patch_size: tuple[int, int, int] | None = None, foreground_probability: float = 0.75, normalization: str = "none", seed: int = 42) -> None:
        self.entries, self.patch_size = entries, patch_size
        self.foreground_probability, self.normalization = foreground_probability, normalization
        self.rng = np.random.default_rng(seed)
        if not 0 <= foreground_probability <= 1: raise ValueError("foreground_probability must be in [0, 1].")
    def __len__(self) -> int: return len(self.entries)
    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        try: import h5py
        except ImportError as exc: raise ImportError("h5py is required to load UA-MT volumes.") from exc
        with h5py.File(entry["h5_path"], "r") as handle:
            if "image" not in handle or "label" not in handle: raise ValueError(f"{entry['h5_path']} must contain image and label.")
            image, label = np.asarray(handle["image"], dtype=np.float32), np.asarray(handle["label"], dtype=np.uint8)
        if image.ndim != 3 or image.shape != label.shape: raise ValueError(f"Invalid image/label shapes: {image.shape}, {label.shape}")
        values = np.unique(label)
        if not np.isin(values, [0, 1]).all(): raise ValueError(f"Label must be binary; found {values.tolist()} in {entry['h5_path']}")
        image = normalize_image(image, self.normalization)
        if self.patch_size: image, label = foreground_or_random_crop(image, label, self.patch_size, self.foreground_probability, self.rng)
        return {"image": torch.from_numpy(image[None]), "label": torch.from_numpy(label[None].astype(np.float32)), "case_id": entry["case_id"]}
