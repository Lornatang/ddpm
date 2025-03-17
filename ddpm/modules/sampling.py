# Copyright (c) Lornatang. All rights reserved.
import torch
from torch import nn
from torch.nn import functional as F_torch

__all__ = [
    "Downsample", "Upsample"
]


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        """Initializes the Downsample block.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.main = nn.Conv2d(channels, channels, (3, 3), stride=(2, 2), padding=(1, 1))

        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass for the Downsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.main(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        """Initializes the upsample block.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.main = nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=(1, 1))

        nn.init.xavier_uniform_(self.main.weight)
        nn.init.zeros_(self.main.bias)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass for the Downsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F_torch.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)

        return x
