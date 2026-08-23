"""Deterministic, case-level splits for the UA-MT HDF5 layout."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping


def find_case_files(data_root: str | Path, filename: str = "mri_norm2.h5") -> list[Path]:
    """Return one HDF5 path per direct case directory, sorted by case ID."""
    root = Path(data_root)
    paths = sorted((path for path in root.glob(f"*/{filename}") if path.is_file()), key=lambda p: p.parent.name)
    if not paths:
        raise FileNotFoundError(f"No {filename!r} files found directly under case folders in {root}")
    ids = [path.parent.name for path in paths]
    if len(ids) != len(set(ids)):
        raise ValueError("Case IDs must be unique.")
    return paths


def create_case_split(
    data_root: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, object]:
    """Create and persist a reproducible split, never splitting a case's patches."""
    fractions = (train_fraction, val_fraction, test_fraction)
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Split fractions must be positive and sum to 1.0.")
    cases = find_case_files(data_root)
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = int(count * train_fraction)
    val_end = train_end + int(count * val_fraction)
    groups = {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}
    result: dict[str, object] = {
        "format_version": 1,
        "data_root": str(Path(data_root).resolve()),
        "seed": seed,
        "fractions": {"train": train_fraction, "val": val_fraction, "test": test_fraction},
        "splits": {name: [{"case_id": path.parent.name, "h5_path": str(path.resolve())} for path in paths] for name, paths in groups.items()},
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_case_split(path: str | Path) -> Mapping[str, object]:
    """Load and minimally validate a split created by :func:`create_case_split`."""
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = split.get("splits")
    if not isinstance(groups, dict) or set(groups) != {"train", "val", "test"}:
        raise ValueError("Split JSON must contain train, val, and test groups.")
    all_ids = [entry["case_id"] for group in groups.values() for entry in group]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A case occurs in more than one split.")
    return split
