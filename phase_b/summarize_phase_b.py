"""
WaveSeg-SSL: Phase B validation summary
=======================================
Ranks every Phase B run (and the Stage 4 boundary reference, lambda_bnd = 0.1)
by best VALIDATION Dice. Uses training_history.json only - never test metrics -
so it is safe to use for choosing lambda_bnd (N2) and the configurations that go
forward to the locked test evaluation.

  python summarize_phase_b.py
"""

import json
from pathlib import Path

CHECKPOINTS_ROOT = Path("/content/drive/MyDrive/waveseg_ssl_checkpoints")
PHASE_B_ROOT = CHECKPOINTS_ROOT / "phase_b"
REFERENCE = CHECKPOINTS_ROOT / "stage4_tuning/exp_wfe_boundary"


VAL_KEYS = ["val_dice", "val_dice_ens", "val_dice_ensemble", "val_ens_dice", "ens_dice", "dice_ens", "val"]


def load_history(hist_path: Path):
    """Return a list of per-epoch dicts; accepts list-of-dicts, {"history": [...]}, or column dicts."""
    data = json.loads(hist_path.read_text())
    if isinstance(data, dict) and "history" in data:
        data = data["history"]
    if isinstance(data, dict):  # column format {"val_dice": [...], ...}
        cols = {k: v for k, v in data.items() if isinstance(v, list)}
        n = max((len(v) for v in cols.values()), default=0)
        data = [{k: v[i] for k, v in cols.items() if i < len(v)} for i in range(n)]
    return data


def find_val_key(rec: dict):
    for k in VAL_KEYS:
        if isinstance(rec.get(k), (int, float)):
            return k
    for k, v in rec.items():  # any numeric "...val...dice..." column
        if "val" in k and "dice" in k and isinstance(v, (int, float)):
            return k
    return None


def summarize(run_dir: Path, name: str):
    hist_path = run_dir / "training_history.json"
    if not hist_path.exists():
        return None
    history = load_history(hist_path)
    if not history:
        print(f"[skip] {name}: empty history")
        return None
    key = find_val_key(history[-1])
    if key is None:
        print(f"[skip] {name}: no validation-Dice column found; keys = {sorted(history[-1])}")
        return None
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    scored = [(i, r) for i, r in enumerate(history) if isinstance(r.get(key), (int, float))]
    i_best, best = max(scored, key=lambda t: t[1][key])
    return {
        "name": name,
        "epochs_done": len(history),
        "best_val": best[key],
        "best_epoch": best.get("epoch", i_best + 1),
        "val_ab": best.get("val_dice_ab", best[key]),
        "lambda": cfg.get("lambda_boundary", "?"),
        "bnd": cfg.get("bnd_loss", "morph"),
        "aug": cfg.get("aug", "none"),
        "cps": cfg.get("cps_mode", "threshold"),
        "third": ("none" if cfg.get("third_branch", "none") == "none"
                  else f"swin+{cfg.get('peft', 'lora')}"),
    }


def main():
    rows = [summarize(REFERENCE, "stage4_boundary (ref)")]
    if PHASE_B_ROOT.exists():
        rows += [summarize(d, d.name) for d in sorted(PHASE_B_ROOT.iterdir()) if d.is_dir()]
    rows = sorted((r for r in rows if r), key=lambda r: r["best_val"], reverse=True)

    print(f"{'Run':<24} | {'Val Dice':<8} | {'Val AB':<8} | {'Ep':<7} | {'bnd':<6} | {'lambda':<6} | {'aug':<6} | {'cps':<9} | third")
    print("-" * 110)
    for r in rows:
        print(f"{r['name']:<24} | {r['best_val']:<8.4f} | {r['val_ab']:<8.4f} | {r['best_epoch']:>3}/{r['epochs_done']:<3} | "
              f"{r['bnd']:<6} | {r['lambda']!s:<6} | {r['aug']:<6} | {r['cps']:<9} | {r['third']}")


if __name__ == "__main__":
    main()
