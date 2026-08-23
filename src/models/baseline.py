"""Configurable supervised MONAI 3D U-Net baseline."""

from __future__ import annotations

from typing import Sequence


def build_baseline_unet(
    channels: Sequence[int] = (16, 32, 64, 128, 256), strides: Sequence[int] = (2, 2, 2, 2), num_res_units: int = 2
):
    """Build a one-channel, binary-logit 3D U-Net (no activation at output)."""
    try:
        from monai.networks.nets import UNet
    except ImportError as exc:  # keeps simple utilities usable without MONAI
        raise ImportError("MONAI is required: install with `pip install monai`." ) from exc
    if len(channels) != len(strides) + 1:
        raise ValueError("channels must have exactly one more item than strides.")
    return UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=tuple(channels), strides=tuple(strides), num_res_units=num_res_units)
