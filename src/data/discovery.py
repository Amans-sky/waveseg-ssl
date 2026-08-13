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

SUPPORTED_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".npy", ".npz", ".h5", ".hdf5")
LABEL_TOKENS = {"label", "labels", "mask", "masks", "seg", "segs", "segmentation", "gt", "truth", "annotation", "annotations"}
# These values come from the user-verified UA-MT example.  They are comparison
# criteria only: every file is still opened and its own keys are inspected.
UA_MT_REFERENCE = {
    "filename": "mri_norm2.h5",
    "keys": ["image", "label"],
    "image_shape": [177, 138, 88],
    "image_dtype": "float32",
    "label_shape": [177, 138, 88],
    "label_dtype": "uint8",
    "label_values": [0, 1],
}


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
    return _suffix(path) in SUPPORTED_SUFFIXES


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


def _h5_dataset_stats(dataset: Any, max_unique_values: int = 256) -> dict[str, Any]:
    """Collect factual HDF5 dataset properties; no image/label names are assumed."""
    if dataset.dtype.kind not in "biufc":
        return {"shape": list(dataset.shape), "dtype": str(dataset.dtype), "numeric": False}
    array = dataset[()]
    stats: dict[str, Any] = {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "numeric": True,
        "min": np.nanmin(array).item(),
        "max": np.nanmax(array).item(),
    }
    # Unique values are particularly useful for a prospective segmentation mask.
    values = np.unique(array)
    if len(values) <= max_unique_values:
        stats["unique_values"] = [value.item() for value in values]
    else:
        stats["unique_values"] = None
        stats["unique_value_count"] = int(len(values))
    return stats


def _infer_h5_roles(datasets: dict[str, dict[str, Any]]) -> tuple[str | None, str | None, list[str]]:
    """Propose roles from inspected names and contents, recording the evidence.

    This is intentionally a proposal rather than a hard-coded UA-MT key mapping.
    """
    notes: list[str] = []
    numeric = {name: info for name, info in datasets.items() if info.get("numeric")}
    image_named = [name for name in numeric if name.lower().split("/")[-1] in {"image", "images", "mri"}]
    label_named = [name for name in numeric if name.lower().split("/")[-1] in {"label", "labels", "mask", "masks", "seg", "segmentation"}]
    image_key = image_named[0] if len(image_named) == 1 else None
    label_key = label_named[0] if len(label_named) == 1 else None
    if image_key:
        notes.append(f"image key proposed from dataset name: {image_key}")
    if label_key:
        notes.append(f"label key proposed from dataset name: {label_key}")
    # If a name alone is not decisive, numeric type and cardinality are evidence,
    # but ambiguity remains visible instead of being silently resolved.
    if label_key is None:
        categorical = [name for name, info in numeric.items() if info.get("unique_values") is not None and len(info["unique_values"]) <= 32]
        if len(categorical) == 1:
            label_key = categorical[0]
            notes.append(f"label key proposed from low unique-value count: {label_key}")
    if image_key is None:
        remaining = [name for name in numeric if name != label_key]
        if len(remaining) == 1:
            image_key = remaining[0]
            notes.append(f"image key proposed as the remaining numeric dataset: {image_key}")
    if image_key is None or label_key is None:
        notes.append("image/label role is ambiguous; inspect the dataset record before using it.")
    return image_key, label_key, notes


def inspect_h5_case(path: Path) -> dict[str, Any]:
    """Inspect every dataset/group in one HDF5 case without modifying it."""
    try:
        import h5py

        datasets: dict[str, dict[str, Any]] = {}
        groups: list[str] = []
        metadata: dict[str, Any] = {}
        with h5py.File(path, "r") as handle:
            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Group):
                    groups.append(name)
                elif isinstance(obj, h5py.Dataset):
                    datasets[name] = _h5_dataset_stats(obj)
                if getattr(obj, "attrs", None):
                    metadata[name or "/"] = {key: _json_value(value) for key, value in obj.attrs.items()}

            visitor("", handle)
            handle.visititems(visitor)
        image_key, label_key, role_notes = _infer_h5_roles(datasets)
        image_shape = datasets.get(image_key, {}).get("shape") if image_key else None
        label_shape = datasets.get(label_key, {}).get("shape") if label_key else None
        reference_deviations = _ua_mt_reference_deviations(path, datasets)
        return {
            "path": str(path.resolve()),
            "case_id": path.parent.name,
            "readable": True,
            "groups": groups,
            "keys": sorted(datasets),
            "datasets": datasets,
            "metadata": metadata,
            "proposed_image_key": image_key,
            "proposed_label_key": label_key,
            "role_inference": role_notes,
            "image_label_spatial_match": image_shape == label_shape if image_shape and label_shape else None,
            "matches_verified_ua_mt_example": not reference_deviations,
            "verified_example_deviations": reference_deviations,
            "error": None,
        }
    except Exception as exc:
        return {"path": str(path.resolve()), "case_id": path.parent.name, "readable": False, "groups": [], "keys": [], "datasets": {}, "metadata": {}, "proposed_image_key": None, "proposed_label_key": None, "role_inference": [], "image_label_spatial_match": None, "matches_verified_ua_mt_example": False, "verified_example_deviations": ["file could not be inspected"], "error": f"{type(exc).__name__}: {exc}"}


def _ua_mt_reference_deviations(path: Path, datasets: dict[str, dict[str, Any]]) -> list[str]:
    """Compare inspected data with the supplied example; never use it to read keys."""
    deviations: list[str] = []
    if path.name != UA_MT_REFERENCE["filename"]:
        deviations.append(f"unexpected HDF5 filename: {path.name}")
    if sorted(datasets) != UA_MT_REFERENCE["keys"]:
        deviations.append(f"dataset keys are {sorted(datasets)}, expected {UA_MT_REFERENCE['keys']}")
    for role in ("image", "label"):
        actual = datasets.get(role)
        if actual is None:
            continue
        for property_name in ("shape", "dtype"):
            expected = UA_MT_REFERENCE[f"{role}_{property_name}"]
            if actual.get(property_name) != expected:
                deviations.append(f"{role} {property_name} is {actual.get(property_name)}, expected {expected}")
    actual_values = datasets.get("label", {}).get("unique_values")
    if actual_values is not None and actual_values != UA_MT_REFERENCE["label_values"]:
        deviations.append(f"label unique values are {actual_values}, expected {UA_MT_REFERENCE['label_values']}")
    return deviations


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def discover_dataset(data_root: str | Path) -> dict[str, Any]:
    """Return a JSON-serialisable report and proposed, explicitly flagged pairs."""
    root = Path(data_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    candidates = sorted((path for path in root.rglob("*") if _is_supported(path)), key=lambda p: str(p).lower())
    h5_paths = [path for path in candidates if _suffix(path) in {".h5", ".hdf5"}]
    h5_cases = [inspect_h5_case(path) for path in h5_paths]
    non_h5_candidates = [path for path in candidates if path not in h5_paths]
    label_paths = [path for path in non_h5_candidates if is_probable_label(path)]
    image_paths = [path for path in non_h5_candidates if path not in label_paths]
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
        if path in h5_paths:
            continue
        record = inspect_volume(path, include_label_values=path in label_paths)
        record_dict = asdict(record)
        record_dict["relative_path"] = str(path.relative_to(root))
        record_dict["kind"] = "probable_label" if path in label_paths else "probable_image"
        records.append(record_dict)

    valid_h5 = [case for case in h5_cases if case["readable"]]
    h5_signatures = Counter(json.dumps({"keys": case["keys"], "datasets": {key: {"shape": value.get("shape"), "dtype": value.get("dtype")} for key, value in case["datasets"].items()}}, sort_keys=True) for case in valid_h5)
    h5_inconsistencies = []
    if len(h5_signatures) > 1:
        h5_inconsistencies.append("Readable HDF5 cases do not all share the same dataset-key/shape/dtype structure.")
    h5_inconsistencies.extend([f"{case['case_id']}: image and label shapes do not match" for case in valid_h5 if case["image_label_spatial_match"] is False])
    h5_inconsistencies.extend([f"{case['case_id']}: {deviation}" for case in valid_h5 for deviation in case["verified_example_deviations"]])
    image_shapes = sorted({tuple(case["datasets"].get("image", {}).get("shape", [])) for case in valid_h5})
    label_shapes = sorted({tuple(case["datasets"].get("label", {}).get("shape", [])) for case in valid_h5})
    image_dtypes = sorted({case["datasets"].get("image", {}).get("dtype") for case in valid_h5})
    label_dtypes = sorted({case["datasets"].get("label", {}).get("dtype") for case in valid_h5})

    return {
        "schema_version": 1,
        "dataset_root": str(root),
        "inspection_only": True,
        "assumptions": [],
        "file_count": len(candidates),
        "formats": dict(sorted(Counter(_suffix(path) for path in candidates).items())),
        "h5_summary": {
            "case_count": len(h5_cases),
            "valid_h5_files": len(valid_h5),
            "invalid_or_unreadable_h5_files": len(h5_cases) - len(valid_h5),
            "cases_with_image_and_label": sum("image" in case["datasets"] and "label" in case["datasets"] for case in valid_h5),
            "cases_with_matching_image_label_shape": sum(case["image_label_spatial_match"] is True for case in valid_h5),
            "all_cases_same_internal_structure": len(h5_signatures) <= 1,
            "structure_signature_count": len(h5_signatures),
            "image_shapes_seen": [list(shape) for shape in image_shapes],
            "label_shapes_seen": [list(shape) for shape in label_shapes],
            "image_dtypes_seen": image_dtypes,
            "label_dtypes_seen": label_dtypes,
            "cases_matching_verified_ua_mt_example": sum(case["matches_verified_ua_mt_example"] for case in valid_h5),
            "reference": UA_MT_REFERENCE,
            "inconsistencies": h5_inconsistencies,
        },
        "h5_cases": h5_cases,
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
