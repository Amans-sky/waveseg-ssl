# WaveSeg-SSL Phase B

Phase B ablations on top of the Stage 4 CPS + WFE + boundary-loss model (LA 2018, 14 labeled / 56 unlabeled / 15 val / 15 locked test).

| File | Purpose |
|---|---|
| `train_phase_b.py` | **The training script.** One flag per ablation; resumable; locked-test evaluation behind `--confirm-frozen` |
| `summarize_phase_b.py` | Ranks all runs by best **validation** Dice (never test) |
| `toy_boundary_loss_study.py` | CPU study of boundary losses on synthetic hazy-boundary phantoms (not needed on Colab) |
| other `.py` | Stage 4 scripts kept for reference (`train_wfe_boundary.py`, `eval_locked_test.py`, `eval_lcc_and_visualize.py`, ...) |

## Flags (`train_phase_b.py`)

| Flag | Values | Item |
|---|---|---|
| `--bnd-loss` | `morph` (Stage 4), `dist` (Kervadec distance loss), `uncert` (uncertainty-rectified boundary loss) | boundary loss |
| `--lambda-bnd` | default `0.1` | N2 |
| `--aug` | `none`, `nnunet` | N3 |
| `--cps-mode` | `threshold` (tau = 0.7), `entropy` (w = 1 - H(p)/log 2) | N4 |
| `--third-branch swin --peft` | `lora`, `adapter` | N5 |
| `--crop` | `fg_random` (default), `center` (overfits - do not use) | |
| `--resume` | continue from `latest.pt` after a disconnect | |

## Google Drive layout the script expects

```
MyDrive/waveseg_ssl_checkpoints/stage3_cps/stage3_cps_split.json      <- locked split (required)
MyDrive/WaveSeg-SSL_Datasets/UA-MT/data/2018LA_Seg_Training Set/...    <- or the UA-MT clone in /content
MyDrive/waveseg_ssl_checkpoints/phase_b/<exp-name>/                    <- outputs (created automatically)
```

## Colab

```python
from google.colab import drive
drive.mount("/content/drive")
!pip install -q "monai==1.6.0" h5py
!git clone -q -b phase-b https://github.com/Amans-sky/waveseg-ssl.git /content/waveseg-ssl
%cd /content/waveseg-ssl/phase_b
!python train_phase_b.py --exp-name n0_repro_bnd_0p1
```
