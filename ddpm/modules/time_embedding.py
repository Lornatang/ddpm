# Copyright (c) Lornatang. All rights reserved.
import math
import torch
from torch import nn

from .activation import Swish

__all__ = [
    "TimeEmbedding",
]


class TimeEmbedding(nn.Module):
    def __init__(self, time_steps: int, channels: int, time_dim: int) -> None:
        """Time embedding module.

        Args:
            time_steps (int): Number of time steps.
            channels (int): Number of channels.
            time_dim (int): Number of time dimensions.
        """
        super().__init__()
        assert channels % 2 == 0, "channels must be even"
        embedding = torch.arange(0, channels, step=2) / channels * math.log(10000)
        embedding = torch.exp(-embedding)
        pos = torch.arange(time_steps).float()
        embedding = pos[:, None] * embedding[None, :]
        assert list(embedding.shape) == [time_steps, channels // 2]
        embedding = torch.stack([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        assert list(embedding.shape) == [time_steps, channels // 2, 2]
        embedding = embedding.view(time_steps, channels)

        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(embedding),
            nn.Linear(channels, time_dim),
            Swish(True),
            nn.Linear(time_dim, time_dim),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor
        """
        return self.time_embedding(x)
