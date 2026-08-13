"""Read-only discovery and metadata inspection for 3D segmentation datasets.

Stage 1 deliberately makes no assumptions about the LA dataset layout, target
spacing, image normalisation, or train/validation/test split.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SUPPORTED_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".npy", ".npz")
LABEL_TOKENS = {"label", "labels", "mask", "masks", "seg", "segs", "segmentation", "gt", "truth", "annotation", "annotations"}


@dataclass(frozen=True)
class VolumeRecord:
    path: str
    relative_path: str
    format: str
    shape: list[int] | None
    spacing: list[float] | None
    orientation: list[str] | None
    label_values: list[float | int] | None
    readable: bool
    error: str | None


def _suffix(path: Path) -> str:
    return ".nii.gz" if path.name.lower().endswith(".nii.gz") else path.suffix.lower()


def _is_supported(path: Path) -> bool:
    return path.is_file() and _suffix(path) in SUPPORTED_SUFFIXES


def _tokens(path: Path) -> set[str]:
    return set(token for token in re.split(r"[^a-z0-9]+", path.stem.lower().replace(".nii", "")) if token)


def is_probable_label(path: Path) -> bool:
    """Classify only from names; the result is reported, never silently trusted."""
    return bool(_tokens(path) & LABEL_TOKENS) or any(part.lower() in LABEL_TOKENS for part in path.parts)


def case_key(path: Path) -> str:
    """Create a conservative name key used only to propose image/label pairs."""
    stem = path.name.lower()
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".npy", ".npz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token and token not in LABEL_TOKENS]
    return "_".join(tokens)


def inspect_volume(path: Path, *, include_label_values: bool) -> VolumeRecord:
    """Read header/array metadata without changing the source file."""
    fmt = _suffix(path)
    try:
        if fmt in {".nii", ".nii.gz"}:
            import nibabel as nib

            image = nib.load(str(path))
            shape = list(image.shape)
            spacing = [float(value) for value in image.header.get_zooms()[: len(shape)]]
            orientation = list(nib.aff2axcodes(image.affine))
            values = _label_values(np.asanyarray(image.dataobj)) if include_label_values else None
        elif fmt in {".npy", ".npz"}:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if isinstance(array, np.lib.npyio.NpzFile):
                keys = list(array.files)
                if len(keys) != 1:
                    raise ValueError(f"NPZ must contain one array; found keys {keys}")
                array = array[keys[0]]
            shape, spacing, orientation = list(array.shape), None, None
            values = _label_values(array) if include_label_values else None
        else:
            import SimpleITK as sitk

            image = sitk.ReadImage(str(path))
            shape = list(reversed(image.GetSize()))  # report as array order: z, y, x
            spacing = [float(value) for value in reversed(image.GetSpacing())]
            orientation = None
            values = _label_values(sitk.GetArrayViewFromImage(image)) if include_label_values else None
        return VolumeRecord(str(path.resolve()), "", fmt, shape, spacing, orientation, values, True, None)
    except Exception as exc:  # Report every unreadable candidate; do not conceal data problems.
        return VolumeRecord(str(path.resolve()), "", fmt, None, None, None, None, False, f"{type(exc).__name__}: {exc}")


def _label_values(array: np.ndarray, max_unique_values: int = 256) -> list[float | int]:
    values = np.unique(array)
    if len(values) > max_unique_values:
        return [f"more_than_{max_unique_values}_unique_values"]  # type: ignore[list-item]
    return [value.item() for value in values]


def discover_dataset(data_root: str | Path) -> dict[str, Any]:
    """Return a JSON-serialisable report and proposed, explicitly flagged pairs."""
    root = Path(data_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    candidates = sorted((path for path in root.rglob("*") if _is_supported(path)), key=lambda p: str(p).lower())
    label_paths = [path for path in candidates if is_probable_label(path)]
    image_paths = [path for path in candidates if path not in label_paths]
    labels_by_key: dict[str, list[Path]] = {}
    for path in label_paths:
        labels_by_key.setdefault(case_key(path), []).append(path)

    proposed_pairs: list[dict[str, str]] = []
    unmatched_images: list[str] = []
    ambiguous_images: list[dict[str, Any]] = []
    used_labels: set[Path] = set()
    for image_path in image_paths:
        matches = labels_by_key.get(case_key(image_path), [])
        if len(matches) == 1:
            label_path = matches[0]
            used_labels.add(label_path)
            proposed_pairs.append({"image": str(image_path.resolve()), "label": str(label_path.resolve()), "match_rule": "normalised filename"})
        elif len(matches) > 1:
            ambiguous_images.append({"image": str(image_path.resolve()), "candidate_labels": [str(path.resolve()) for path in matches]})
        else:
            unmatched_images.append(str(image_path.resolve()))

    records = []
    for path in candidates:
        record = inspect_volume(path, include_label_values=path in label_paths)
        record_dict = asdict(record)
        record_dict["relative_path"] = str(path.relative_to(root))
        record_dict["kind"] = "probable_label" if path in label_paths else "probable_image"
        records.append(record_dict)

    return {
        "schema_version": 1,
        "dataset_root": str(root),
        "inspection_only": True,
        "assumptions": [],
        "file_count": len(candidates),
        "formats": dict(sorted(Counter(_suffix(path) for path in candidates).items())),
        "records": records,
        "proposed_pairs": proposed_pairs,
        "unmatched_images": unmatched_images,
        "unmatched_labels": [str(path.resolve()) for path in label_paths if path not in used_labels],
        "ambiguous_images": ambiguous_images,
        "read_errors": [record for record in records if not record["readable"]],
    }


def write_manifest(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output


def iter_readable_pairs(report: dict[str, Any]) -> Iterable[dict[str, str]]:
    unreadable = {record["path"] for record in report["read_errors"]}
    yield from (pair for pair in report["proposed_pairs"] if pair["image"] not in unreadable and pair["label"] not in unreadable)
