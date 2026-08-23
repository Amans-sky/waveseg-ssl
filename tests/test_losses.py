import torch
import pytest
from src.losses.supervised import DiceBCEWithLogitsLoss

def test_loss_is_finite_and_accepts_binary_target():
 logits=torch.randn(2,1,8,8,8,requires_grad=True); target=torch.randint(0,2,logits.shape).float()
 loss=DiceBCEWithLogitsLoss()(logits,target); loss.backward(); assert torch.isfinite(loss)

def test_loss_rejects_mismatched_shapes():
 with pytest.raises(ValueError): DiceBCEWithLogitsLoss()(torch.zeros(1,1,2,2,2),torch.zeros(1,2,2,2))
