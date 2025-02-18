# Copyright (c) Lornatang. All rights reserved.
import math

import torch
from torch import nn

__all__ = [
    "SinusoidalPosEmbedding",
]


class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, dim: int, theta: int = 10000) -> None:
        """Initializes the SinusoidalPosEmbedding module.

        Args:
            dim (int): Dimension of the embedding.
            theta (int, optional): Theta value for the sinusoidal position embedding. Defaults to 10000.
        """
        super().__init__()
        self.dim: int = dim
        self.theta: int = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the sinusoidal position embedding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sinusoidal position embedding tensor.
        """
        half_dim = self.dim // 2
        embedding = math.log(self.theta) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=x.device) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)

        return embedding
