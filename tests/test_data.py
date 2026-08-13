from pathlib import Path

import numpy as np
import pytest

from src.data.discovery import discover_dataset, write_manifest
from src.data.transforms import PreprocessingNotApprovedError, build_preprocessing_transforms


def test_discovery_proposes_a_filename_matched_pair(tmp_path: Path) -> None:
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    np.save(images / "case_001.npy", np.zeros((4, 5, 6), dtype=np.float32))
    np.save(labels / "case_001_label.npy", np.zeros((4, 5, 6), dtype=np.uint8))
    report = discover_dataset(tmp_path)
    assert report["file_count"] == 2
    assert len(report["proposed_pairs"]) == 1
    assert report["records"][1]["label_values"] == [0]
    manifest = write_manifest(report, tmp_path / "manifest.json")
    assert manifest.exists()


def test_preprocessing_is_blocked_in_stage_1() -> None:
    with pytest.raises(PreprocessingNotApprovedError):
        build_preprocessing_transforms()
