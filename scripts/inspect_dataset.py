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


def save_representative_slices(report: dict, output_path: Path) -> None:
    pairs = report["proposed_pairs"]
    if not pairs:
        print("No unambiguous image/label pair available; no slice preview was created.")
        return
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
    axes[0].set_title(f"Image (axis={axis}, index={index})")
    axes[1].imshow(label_slice.T, cmap="tab20", origin="lower", interpolation="nearest")
    axes[1].set_title("Label")
    axes[2].imshow(image_slice.T, cmap="gray", origin="lower")
    axes[2].imshow(np.ma.masked_where(label_slice.T == 0, label_slice.T), cmap="tab20", alpha=0.45, origin="lower")
    axes[2].set_title("Overlay")
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
    print(f"Proposed pairs: {len(report['proposed_pairs'])}; unmatched images: {len(report['unmatched_images'])}; unmatched labels: {len(report['unmatched_labels'])}; ambiguous: {len(report['ambiguous_images'])}; unreadable: {len(report['read_errors'])}")
    print("Inspection complete. No preprocessing, split, or training was performed.")


if __name__ == "__main__":
    main()
