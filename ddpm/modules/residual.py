# Copyright (c) Lornatang. All rights reserved.
import torch
from torch import nn

from .activation import Swish
from .attention import Attention

__all__ = [
    "Residual",
]


class Residual(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_dim: int,
            attn: bool = False,
            dropout: float = 0,
    ) -> None:
        """Initializes the Residual block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            time_dim (int): Dimension of the time embedding.
            attn (bool, optional): Whether to use attention. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to 0.
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(True),
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.time_embedding_proj = nn.Sequential(
            Swish(True),
            nn.Linear(time_dim, out_channels),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            Swish(True),
            nn.Dropout(dropout, True),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1), stride=(1, 1), padding=(0, 0)) if in_channels != out_channels else nn.Identity()
        self.attn = Attention(out_channels) if attn else nn.Identity()

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Residual module.

        Args:
            x (torch.Tensor): Input tensor.
            time_embedding (Optional[torch.Tensor]): Time embedding tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual block operations.
        """
        out = self.block1(x)
        out += self.time_embedding_proj(time_embedding)[:, :, None, None]
        out = self.block2(out)

        out = out + self.shortcut(x)
        out = self.attn(out)

        return out
