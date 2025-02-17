# Copyright (c) Lornatang. All rights reserved.
from einops.layers.torch import Rearrange
from torch import nn

__all__ = [
    "Downsample",
]


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
