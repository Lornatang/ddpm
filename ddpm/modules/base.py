# Copyright (c) Lornatang. All rights reserved.
from typing import Optional, Tuple

import torch
from torch import nn

from .activation import RMSNorm

__all__ = [
    "Block",
]


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.) -> None:
        """Initializes the Block module.

        Args:
            in_channels (int): Dimension of the input.
            out_channels (int): Dimension of the output.
            dropout (float, optional): Dropout rate. Defaults to 0.
        """
        super().__init__()
        self.proj: nn.Module = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.norm: nn.Module = RMSNorm(out_channels)
        self.act: nn.Module = nn.SiLU(True)
        self.dropout: nn.Module = nn.Dropout(dropout, True)

    def forward(self, x: torch.Tensor, scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass for the Block module.

        Args:
            x (torch.Tensor): Input tensor.
            scale_shift (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Scale and shift tensors. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying the block operations.
        """
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)
