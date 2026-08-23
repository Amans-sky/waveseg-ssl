"""Train Stage 2 after `create_split.py` and `test_overfit.py` pass."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.datasets import UAMTH5Dataset
from src.data.splits import load_case_split
from src.models.baseline import build_baseline_unet
from src.losses.supervised import DiceBCEWithLogitsLoss
from src.training.engine import train_epoch, validate
from src.training.checkpointing import load_checkpoint, save_checkpoint

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/stage2_supervised.yaml'); p.add_argument('--resume'); a=p.parse_args()
 c=yaml.safe_load(Path(a.config).read_text()); split=load_case_split(c['split_path'])['splits']; d=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 patch=tuple(c['patch_size']); common=dict(normalization=c['normalization'], foreground_probability=c['foreground_probability'], seed=c['seed'])
 train=DataLoader(UAMTH5Dataset(split['train'],patch_size=patch,**common),batch_size=c['training']['batch_size'],shuffle=True,num_workers=c['training']['num_workers'],pin_memory=d.type=='cuda')
 val=DataLoader(UAMTH5Dataset(split['val'],patch_size=None,**common),batch_size=1,num_workers=c['training']['num_workers'])
 model=build_baseline_unet(**c['model']).to(d); opt=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate']); scaler=torch.cuda.amp.GradScaler(enabled=d.type=='cuda'); start=0; best=-1.
 if a.resume:
  state=load_checkpoint(a.resume,model=model,optimizer=opt,scaler=scaler,device=d); start=state['epoch']+1; best=state['best_val_dice']
 out=Path(c['training']['checkpoint_dir'])
 for epoch in range(start,c['training']['epochs']):
  loss=train_epoch(model,train,opt,DiceBCEWithLogitsLoss(),d,scaler); dice=validate(model,val,d,patch)
  print(f'epoch={epoch} train_loss={loss:.5f} val_dice={dice:.5f}')
  save_checkpoint(out/'last.pt',model=model,optimizer=opt,scaler=scaler,epoch=epoch,config=c,best_val_dice=max(best,dice))
  if dice>best: best=dice; save_checkpoint(out/'best.pt',model=model,optimizer=opt,scaler=scaler,epoch=epoch,config=c,best_val_dice=best)
if __name__ == '__main__': main()
