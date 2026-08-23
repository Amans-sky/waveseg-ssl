from __future__ import annotations
from pathlib import Path
import torch

def save_checkpoint(path: str | Path, *, model, optimizer, scaler, epoch: int, config: dict, best_val_dice: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict() if scaler else None, "epoch": epoch, "config": config, "best_val_dice": best_val_dice}, path)

def load_checkpoint(path: str | Path, *, model, optimizer=None, scaler=None, device="cpu") -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer: optimizer.load_state_dict(state["optimizer_state"])
    if scaler and state.get("scaler_state"): scaler.load_state_dict(state["scaler_state"])
    return state
