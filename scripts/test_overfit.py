"""Required Stage 2 gate: fit one/two cases, then report training-volume Dice."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader, RandomSampler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.datasets import UAMTH5Dataset
from src.data.splits import load_case_split
from src.models.baseline import build_baseline_unet
from src.losses.supervised import DiceBCEWithLogitsLoss
from src.training.engine import train_epoch, validate
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stage2_supervised.yaml');p.add_argument('--cases',type=int,default=1);p.add_argument('--epochs',type=int,default=100);p.add_argument('--steps-per-epoch',type=int,default=20);p.add_argument('--target-dice',type=float,default=.90)
if __name__=='__main__':
 a=p.parse_args(); c=yaml.safe_load(Path(a.config).read_text()); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); entries=load_case_split(c['split_path'])['splits']['train'][:a.cases]; patch=tuple(c['patch_size']); ds=UAMTH5Dataset(entries,patch_size=patch,normalization=c['normalization'],foreground_probability=1.0,seed=c['seed']); loader=DataLoader(ds,batch_size=1,sampler=RandomSampler(ds,replacement=True,num_samples=a.steps_per_epoch)); model=build_baseline_unet(**c['model']).to(d); opt=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate']*10); scaler=torch.cuda.amp.GradScaler(enabled=d.type=='cuda')
 for e in range(a.epochs): print(f'epoch={e} loss={train_epoch(model,loader,opt,DiceBCEWithLogitsLoss(),d,scaler):.5f}')
 dice=validate(model,DataLoader(UAMTH5Dataset(entries,normalization=c['normalization']),batch_size=1),d,patch); print(f'overfit_dice={dice:.5f} target={a.target_dice:.5f}'); raise SystemExit(0 if dice>=a.target_dice else 1)
