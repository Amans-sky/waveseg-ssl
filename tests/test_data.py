from pathlib import Path

import numpy as np
import pytest

from src.data.discovery import _is_supported, discover_dataset, write_manifest
from src.data.transforms import PreprocessingNotApprovedError, build_preprocessing_transforms
from src.data.datasets import UAMTH5Dataset
from src.data.splits import create_case_split


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


def test_h5_file_is_supported(tmp_path: Path) -> None:
    h5_path = tmp_path / "case_001" / "mri_norm2.h5"
    h5_path.parent.mkdir()
    h5_path.touch()
    assert _is_supported(h5_path) is True


def test_discovery_inspects_ua_mt_style_h5_case_without_hard_coded_keys(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py", reason="HDF5 inspection requires h5py")
    case = tmp_path / "R8ER97O9UUN77C02VE2J"
    case.mkdir()
    with h5py.File(case / "mri_norm2.h5", "w") as handle:
        handle.create_dataset("image", data=np.linspace(-1, 1, 24, dtype=np.float32).reshape(2, 3, 4))
        handle.create_dataset("label", data=np.array([[[0, 1, 1, 0]]], dtype=np.uint8))
        handle.attrs["spacing"] = [1.0, 1.0, 1.0]
    report = discover_dataset(tmp_path)
    assert report["file_count"] == 1
    assert report["h5_summary"]["case_count"] == 1
    assert report["h5_summary"]["valid_h5_files"] == 1
    assert report["h5_summary"]["cases_with_image_and_label"] == 1
    inspected = report["h5_cases"][0]
    assert inspected["keys"] == ["image", "label"]
    assert inspected["datasets"]["image"]["shape"] == [2, 3, 4]
    assert inspected["datasets"]["image"]["dtype"] == "float32"
    assert inspected["datasets"]["label"]["unique_values"] == [0, 1]
    assert inspected["proposed_image_key"] == "image"
    assert inspected["proposed_label_key"] == "label"
    assert inspected["image_label_spatial_match"] is False
    assert inspected["matches_verified_ua_mt_example"] is False
    assert any("image shape" in issue for issue in inspected["verified_example_deviations"])


def test_verified_ua_mt_example_is_reported_as_consistent(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py", reason="HDF5 inspection requires h5py")
    case = tmp_path / "R8ER97O9UUN77C02VE2J"
    case.mkdir()
    image = np.zeros((177, 138, 88), dtype=np.float32)
    label = np.zeros((177, 138, 88), dtype=np.uint8)
    label[0, 0, 0] = 1
    with h5py.File(case / "mri_norm2.h5", "w") as handle:
        handle.create_dataset("image", data=image)
        handle.create_dataset("label", data=label)
    report = discover_dataset(tmp_path)
    assert report["h5_summary"]["cases_matching_verified_ua_mt_example"] == 1
    assert report["h5_cases"][0]["verified_example_deviations"] == []


def test_stage2_split_and_lazy_h5_loader(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    for index in range(10):
        case = tmp_path / f"case_{index:02d}"; case.mkdir()
        with h5py.File(case / "mri_norm2.h5", "w") as handle:
            handle.create_dataset("image", data=np.zeros((5, 6, 7), dtype=np.float32))
            handle.create_dataset("label", data=np.zeros((5, 6, 7), dtype=np.uint8))
    split = create_case_split(tmp_path, tmp_path / "split.json", seed=7)
    assert [len(split["splits"][name]) for name in ("train", "val", "test")] == [7, 1, 2]
    item = UAMTH5Dataset(split["splits"]["train"], patch_size=(8, 8, 8))[0]
    assert item["image"].shape == item["label"].shape == (1, 8, 8, 8)
