"""
WaveSeg-SSL: Boundary loss behaviour on hazy boundaries (controlled synthetic study)
=================================================================================
Question: does the soft-morphological boundary loss hurt when the boundary is hazy,
and does the uncertainty-rectified boundary loss behave better?

Synthetic 3D phantoms reproduce the failure mode in a setting where the true
boundary is known exactly:
  - one half of every object has a sharp edge, the other half a hazy edge
    (strong blur, lower contrast, more noise) - like LGE-MRI scar/vein borders
  - training labels in the hazy half are shifted inwards/outwards by up to 2 voxels
    per case (inter-annotator variability on ambiguous borders); the sharp half is exact
  - bright distractor blobs away from the object provoke far false positives (HD95)
  - test metrics are computed against the clean, true shapes

All variants train the same two-branch model (two small 3D UNets, mean ensemble),
same data, same seeds, same lambda; only the boundary term changes. The loss
functions are imported from train_phase_b.py, so this tests the exact code that
runs on Colab. Runs on CPU in ~15-30 min.

  python toy_boundary_loss_study.py [--seeds 3] [--steps 500] [--lam 0.5]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import compute_hausdorff_distance
from monai.networks.nets import UNet
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter

from train_phase_b import (
    boundary_dice_loss, dice_loss, distance_boundary_loss, surface_distance_map, uncertainty_boundary_loss,
)

SHAPE = (48, 48, 40)
VARIANTS = ["none", "morph", "dist", "uncert"]


def make_case(seed: int, noisy_label: bool):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in SHAPE], indexing="ij")
    c = np.array(SHAPE) / 2 + rng.uniform(-3, 3, 3)
    r = np.array([13, 12, 10]) * rng.uniform(0.85, 1.15, 3)
    # Smoothly perturbed ellipsoid
    wobble = gaussian_filter(rng.normal(0, 1, SHAPE), 4)
    wobble = 0.12 * wobble / (np.abs(wobble).max() + 1e-6)
    rho = np.sqrt(((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2)
    truth = rho <= 1.0 + wobble

    hazy = xx > c[2]  # right half of the object has the hazy boundary
    obj = truth.astype(np.float32)
    sharp_img = gaussian_filter(obj, 0.7)
    hazy_img = 0.65 * gaussian_filter(obj, 3.0)
    image = np.where(hazy, hazy_img, sharp_img)

    # Distractors: 2 bright blobs outside the object
    for _ in range(2):
        while True:
            p = np.array([rng.uniform(5, s - 5) for s in SHAPE])
            if not truth[tuple(p.astype(int))] and np.linalg.norm((p - c) / r) > 1.5:
                break
        blob = ((zz - p[0]) ** 2 + (yy - p[1]) ** 2 + (xx - p[2]) ** 2) <= rng.uniform(2.5, 4) ** 2
        image = image + 0.8 * gaussian_filter(blob.astype(np.float32), 1.0)

    noise_sd = np.where(hazy, 0.22, 0.12)
    image = (image + rng.normal(0, 1, SHAPE) * noise_sd).astype(np.float32)

    label = truth.copy()
    if noisy_label:
        k = int(rng.integers(-2, 3))  # annotator shifts the hazy border by k voxels
        if k > 0:
            moved = binary_dilation(truth, iterations=k)
        elif k < 0:
            moved = binary_erosion(truth, iterations=-k)
        else:
            moved = truth
        label = np.where(hazy, moved, truth)
    return image, label.astype(np.float32), truth.astype(np.float32), hazy


def surface(mask):
    return mask & ~binary_erosion(mask)


def region_asd(pred, truth, region):
    """Mean distance from predicted surface voxels inside `region` to the true surface."""
    ps = surface(pred) & region
    if not ps.any():
        return float("nan")
    dt = distance_transform_edt(~surface(truth))
    return float(dt[ps].mean())


def new_branch():
    return UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(8, 16, 32, 64),
                strides=(2, 2, 2), num_res_units=1)


def train_and_eval(variant, seed, train_set, test_set, steps, lam, band):
    torch.manual_seed(seed)
    branches = [new_branch(), new_branch()]
    params = [p for b in branches for p in b.parameters()]
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)

    for step in range(steps):
        idx = rng.choice(len(train_set), 2, replace=False)
        x = torch.stack([train_set[i]["x"] for i in idx])
        y = torch.stack([train_set[i]["y"] for i in idx])
        d = torch.stack([train_set[i]["d"] for i in idx])
        logits = [b(x) for b in branches]
        loss = sum(F.binary_cross_entropy_with_logits(lg, y) + dice_loss(lg, y) for lg in logits) / 2
        if variant == "morph":
            loss = loss + lam * sum(boundary_dice_loss(lg, y) for lg in logits) / 2
        elif variant == "dist":
            loss = loss + lam * sum(distance_boundary_loss(lg, y, d) for lg in logits) / 2
        elif variant == "uncert":
            consensus = torch.stack([torch.sigmoid(lg) for lg in logits]).mean(0).detach()
            loss = loss + lam * sum(uncertainty_boundary_loss(lg, y, d, consensus, band)[0] for lg in logits) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()

    for b in branches:
        b.eval()
    rows = []
    with torch.no_grad():
        for case in test_set:
            prob = torch.stack([torch.sigmoid(b(case["x"][None])) for b in branches]).mean(0)
            pred_t = (prob >= 0.5).float()
            truth_t = case["truth"][None]
            pred, truth = pred_t[0, 0].numpy().astype(bool), truth_t[0, 0].numpy().astype(bool)
            inter = (pred & truth).sum()
            dice = 2 * inter / (pred.sum() + truth.sum() + 1e-6)
            hd = float(compute_hausdorff_distance(pred_t, truth_t, include_background=True, percentile=95.0)[0, 0])
            hazy = case["hazy"]
            rows.append({
                "dice": float(dice), "hd95": hd,
                "asd_hazy": region_asd(pred, truth, hazy), "asd_sharp": region_asd(pred, truth, ~hazy),
                "precision": float(inter / max(pred.sum(), 1)), "recall": float(inter / max(truth.sum(), 1)),
            })
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--band", type=float, default=2.0)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--out", type=str, default="toy_boundary_loss_results.json")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    def pack(seed, noisy):
        img, lab, truth, hazy = make_case(seed, noisy)
        return {"x": torch.from_numpy(img[None]), "y": torch.from_numpy(lab[None]),
                "d": torch.from_numpy(surface_distance_map(lab)[None]),
                "truth": torch.from_numpy(truth[None]), "hazy": hazy}

    train_set = [pack(1000 + i, True) for i in range(args.n_train)]
    test_set = [pack(5000 + i, False) for i in range(args.n_test)]

    results = {v: [] for v in VARIANTS}
    for seed in range(args.seeds):
        for v in VARIANTS:
            m = train_and_eval(v, seed, train_set, test_set, args.steps, args.lam, args.band)
            results[v].append(m)
            print(f"seed {seed} | {v:<6} | Dice {m['dice']:.4f} | HD95 {m['hd95']:6.2f} | "
                  f"ASD hazy {m['asd_hazy']:.3f} | ASD sharp {m['asd_sharp']:.3f} | "
                  f"P {m['precision']:.3f} R {m['recall']:.3f}", flush=True)

    print("\n" + "=" * 96)
    print(f"MEAN +/- SD over {args.seeds} seeds  (lambda={args.lam}, {args.steps} steps, "
          f"{args.n_train} train / {args.n_test} test phantoms)")
    print(f"{'Loss':<8} | {'Dice':<15} | {'HD95 (vox)':<15} | {'ASD hazy':<15} | {'ASD sharp':<15}")
    print("-" * 96)
    summary = {}
    for v in VARIANTS:
        s = {k: (float(np.mean([r[k] for r in results[v]])), float(np.std([r[k] for r in results[v]])))
             for k in results[v][0]}
        summary[v] = s
        print(f"{v:<8} | {s['dice'][0]:.4f} +/- {s['dice'][1]:.4f} | {s['hd95'][0]:6.2f} +/- {s['hd95'][1]:5.2f} | "
              f"{s['asd_hazy'][0]:.3f} +/- {s['asd_hazy'][1]:.3f} | {s['asd_sharp'][0]:.3f} +/- {s['asd_sharp'][1]:.3f}")
    Path(args.out).write_text(json.dumps({"args": vars(args), "runs": results, "summary": summary}, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
