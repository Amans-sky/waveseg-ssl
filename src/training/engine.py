"""Small, explicit supervised training engine with full-volume validation."""
from __future__ import annotations
import torch
from torch.cuda.amp import GradScaler, autocast
from .metrics import binary_dice_from_logits

def train_epoch(model, loader, optimizer, loss_fn, device: torch.device, scaler: GradScaler | None = None) -> float:
    model.train(); total = 0.0
    for batch in loader:
        image, label = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            loss = loss_fn(model(image), label)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else: loss.backward(); optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, device: torch.device, roi_size: tuple[int, int, int], sw_batch_size: int = 1) -> float:
    try: from monai.inferers import sliding_window_inference
    except ImportError as exc: raise ImportError("MONAI is required for sliding-window validation.") from exc
    model.eval(); scores = []
    for batch in loader:
        image, label = batch["image"].to(device), batch["label"].to(device)
        logits = sliding_window_inference(image, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=0.25)
        scores.append(binary_dice_from_logits(logits, label).item())
    return sum(scores) / max(1, len(scores))
