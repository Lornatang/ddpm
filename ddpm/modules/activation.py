# Copyright (c) Lornatang. All rights reserved.
import torch
from torch import nn
from torch.nn import functional as F_torch

__all__ = [
    "RMSNorm",
]


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        """Initialize the RMSNorm module.

        Args:
            dim (int): The dimension of the input tensor.
        """
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the RMSNorm module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized and scaled tensor.
        """
        return F_torch.normalize(x, dim=1) * self.g * self.scale
