# Copyright (c) Lornatang. All rights reserved.
import torch
from torch import nn

__all__ = [
    "Swish",
]


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        """Swish activation function.

        Args:
            inplace (bool, optional): Whether to do the operation in-place. Defaults to False.
        """
        super().__init__()
        self.inplace: bool = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x * torch.sigmoid(x) if self.inplace else x * torch.sigmoid(x).clone()
