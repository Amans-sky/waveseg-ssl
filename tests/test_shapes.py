import pytest
torch = pytest.importorskip('torch')
from src.data.transforms import foreground_or_random_crop
from src.models.baseline import build_baseline_unet
import numpy as np

def test_crop_has_requested_shape():
 im=np.zeros((8,9,10),np.float32); lb=im.astype(np.uint8); lb[1,1,1]=1
 x,y=foreground_or_random_crop(im,lb,(12,12,12),1,np.random.default_rng(0))
 assert x.shape == y.shape == (12,12,12)

def test_model_forward_shape():
 pytest.importorskip('monai')
 m=build_baseline_unet(channels=(4,8),strides=(2,)); x=torch.randn(1,1,16,16,16)
 assert m(x).shape == x.shape
