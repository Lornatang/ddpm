# Copyright (c) Lornatang. All rights reserved.
from typing import List, Tuple

import torch
from torch import nn

from ddpm.modules.activation import Swish
from ddpm.modules.residual import Residual
from ddpm.modules.sampling import Downsample, Upsample
from ddpm.modules.time_embedding import TimeEmbedding

__all__ = [
    "UNet",
]


class UNet(nn.Module):
    def __init__(
            self,
            time_steps: int = 1000,
            in_channels: int = 3,
            channels: int = 128,
            channels_multipliers: Tuple[int, ...] = (1, 2, 2, 2),
            attn_multipliers: List[int] = None,
            num_res_blocks: int = 2,
            init_weights: bool = True,
            dropout: float = 0,
    ) -> None:
        """Initializes the UNet module.

        Args:
            time_steps (int, optional): Number of time steps. Defaults to 1000.
            in_channels (int): Number of input channels. Defaults to 3.
            channels (int, optional): Number of channels. Defaults to 128.
            channels_multipliers (Tuple[int, ...]): Multipliers for the channels at each stage. Defaults to (1, 2, 2, 2).
            attn_multipliers (List[int]): Multipliers for the attention. Defaults to ``None``.
            num_res_blocks (int, optional): Number of residual blocks. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels

        if attn_multipliers is None:
            attn_multipliers = [1]
        assert all([i < len(channels_multipliers) for i in attn_multipliers])

        time_dim = channels * 4
        self.time_embedding = TimeEmbedding(time_steps, channels, time_dim)

        self.head = nn.Conv2d(in_channels, channels, (3, 3), stride=(1, 1), padding=(1, 1))

        self.down_sampling = nn.ModuleList()
        channel_list = [channels]
        current_channel = channels
        for i, mult in enumerate(channels_multipliers):
            out_channels = channels * mult
            for _ in range(num_res_blocks):
                self.down_sampling.append(
                    Residual(
                        current_channel,
                        out_channels,
                        time_dim,
                        attn=(i in attn_multipliers),
                        dropout=dropout,
                    )
                )
                current_channel = out_channels
                channel_list.append(current_channel)
            if i != len(channels_multipliers) - 1:
                self.down_sampling.append(Downsample(current_channel))
                channel_list.append(current_channel)

        self.residual_blocks = nn.ModuleList([
            Residual(current_channel, current_channel, time_dim, attn=True, dropout=dropout),
            Residual(current_channel, current_channel, time_dim, attn=False, dropout=dropout),
        ])

        self.up_sampling = nn.ModuleList([])
        for i, mult in reversed(list(enumerate(channels_multipliers))):
            out_channels = channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_sampling.append(
                    Residual(
                        channel_list.pop() + current_channel,
                        out_channels,
                        time_dim,
                        attn=(i in attn_multipliers),
                        dropout=dropout,
                    )
                )
                current_channel = out_channels
            if i != 0:
                self.up_sampling.append(Upsample(current_channel))
        assert len(channel_list) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, current_channel),
            Swish(True),
            nn.Conv2d(current_channel, in_channels, (3, 3), stride=(1, 1), padding=(1, 1))
        )

        if init_weights:
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
            nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        """Forward pass for the UNet module.

        Args:
            x (torch.Tensor): Input tensor.
            time_step (torch.Tensor): Time embedding tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the UNet.
        """
        time_embedding = self.time_embedding(time_step)
        x = self.head(x)

        x_list = [x]
        for layer in self.down_sampling:
            x = layer(x, time_embedding)
            x_list.append(x)

        for layer in self.residual_blocks:
            x = layer(x, time_embedding)

        for layer in self.up_sampling:
            if isinstance(layer, Residual):
                x = torch.cat([x, x_list.pop()], dim=1)
            x = layer(x, time_embedding)

        x = self.tail(x)
        assert len(x_list) == 0

        return x
