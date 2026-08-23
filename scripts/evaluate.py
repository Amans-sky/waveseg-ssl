from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.datasets import UAMTH5Dataset
from src.data.splits import load_case_split
from src.models.baseline import build_baseline_unet
from src.training.engine import validate
from src.training.checkpointing import load_checkpoint
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stage2_supervised.yaml');p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=('val','test'),default='test')
if __name__=='__main__':
 a=p.parse_args();c=yaml.safe_load(Path(a.config).read_text());d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=build_baseline_unet(**c['model']).to(d);load_checkpoint(a.checkpoint,model=m,device=d); entries=load_case_split(c['split_path'])['splits'][a.split]; loader=DataLoader(UAMTH5Dataset(entries,normalization=c['normalization']),batch_size=1);print(f'{a.split}_dice={validate(m,loader,d,tuple(c["patch_size"])):.5f}')
