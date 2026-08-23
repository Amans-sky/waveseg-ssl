from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import create_case_split
p = argparse.ArgumentParser(); p.add_argument('--data-root', required=True); p.add_argument('--output', default='data/splits/stage2_split.json'); p.add_argument('--seed', type=int, default=42); p.add_argument('--train', type=float, default=.70); p.add_argument('--val', type=float, default=.15); p.add_argument('--test', type=float, default=.15)
if __name__ == '__main__':
 a=p.parse_args(); s=create_case_split(a.data_root,a.output,seed=a.seed,train_fraction=a.train,val_fraction=a.val,test_fraction=a.test); print({k:len(v) for k,v in s['splits'].items()})
