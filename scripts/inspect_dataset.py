"""Create a reproducible, read-only LA-dataset inspection manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.discovery import discover_dataset, write_manifest  # noqa: E402


def _load_array(path: Path) -> np.ndarray:
    if path.name.lower().endswith((".nii", ".nii.gz")):
        import nibabel as nib

        return np.asanyarray(nib.load(str(path)).dataobj)
    if path.suffix.lower() in {".npy", ".npz"}:
        array = np.load(path, allow_pickle=False)
        return array[array.files[0]] if isinstance(array, np.lib.npyio.NpzFile) else array
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _load_h5_array(path: Path, key: str) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle[key])


def save_representative_slices(report: dict, output_path: Path) -> None:
    h5_cases = report.get("h5_cases", [])
    h5_case = next((case for case in h5_cases if case["readable"] and case["proposed_image_key"] and case["proposed_label_key"]), None)
    if h5_case:
        image = _load_h5_array(Path(h5_case["path"]), h5_case["proposed_image_key"])
        label = _load_h5_array(Path(h5_case["path"]), h5_case["proposed_label_key"])
        title_prefix = f"H5 case {h5_case['case_id']}"
    else:
        title_prefix = ""
        image = label = None
    pairs = report["proposed_pairs"]
    if image is None and not pairs:
        print("No unambiguous image/label pair available; no slice preview was created.")
        return
    if image is None:
        pair = pairs[0]
        image, label = _load_array(Path(pair["image"])), _load_array(Path(pair["label"]))
    if image.shape != label.shape:
        print(f"Preview skipped: image shape {image.shape} differs from label shape {label.shape}.")
        return
    axis = int(np.argmin(image.shape))
    index = image.shape[axis] // 2
    image_slice, label_slice = np.take(image, index, axis=axis), np.take(label, index, axis=axis)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"{title_prefix} MRI (axis={axis}, index={index})")
    axes[1].imshow(label_slice.T, cmap="tab20", origin="lower", interpolation="nearest")
    axes[1].set_title("Segmentation mask")
    axes[2].imshow(image_slice.T, cmap="gray", origin="lower")
    axes[2].imshow(np.ma.masked_where(label_slice.T == 0, label_slice.T), cmap="tab20", alpha=0.45, origin="lower")
    axes[2].set_title("MRI + mask overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Mounted dataset directory; never modified.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "manifests"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    report = discover_dataset(args.data_root)
    manifest = write_manifest(report, output_dir / "dataset_manifest.json")
    save_representative_slices(report, output_dir / "representative_slices.png")
    print(f"Manifest: {manifest}")
    print(f"Files: {report['file_count']}; formats: {report['formats']}")
    h5 = report["h5_summary"]
    print(f"HDF5 cases: {h5['case_count']}; valid: {h5['valid_h5_files']}; invalid/unreadable: {h5['invalid_or_unreadable_h5_files']}")
    print(f"Cases containing both image and label: {h5['cases_with_image_and_label']}; matching image/label shapes: {h5['cases_with_matching_image_label_shape']}")
    print(f"Image shapes: {h5['image_shapes_seen']}; label shapes: {h5['label_shapes_seen']}")
    print(f"Image dtypes: {h5['image_dtypes_seen']}; label dtypes: {h5['label_dtypes_seen']}")
    print(f"Cases matching the supplied verified example: {h5['cases_matching_verified_ua_mt_example']}")
    if report["h5_cases"]:
        first_h5 = next((case for case in report["h5_cases"] if case["readable"]), None)
        if first_h5:
            print(f"HDF5 keys (first valid case): {first_h5['keys']}")
            print(f"Proposed image key: {first_h5['proposed_image_key']}; proposed label key: {first_h5['proposed_label_key']}")
    if h5["inconsistencies"]:
        print("HDF5 inconsistencies:")
        for item in h5["inconsistencies"]:
            print(f"- {item}")
    print(f"Proposed pairs: {len(report['proposed_pairs'])}; unmatched images: {len(report['unmatched_images'])}; unmatched labels: {len(report['unmatched_labels'])}; ambiguous: {len(report['ambiguous_images'])}; unreadable: {len(report['read_errors'])}")
    print("Inspection complete. No preprocessing, split, or training was performed.")


if __name__ == "__main__":
    main()
