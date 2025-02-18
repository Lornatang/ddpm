# Copyright (c) Lornatang. All rights reserved.
from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn, Module

from ddpm.modules.attention import Attention, LinearAttention
from ddpm.modules.downsampling import Downsample
from ddpm.modules.residual import Residual
from ddpm.modules.sinusoidal_pos_embedding import SinusoidalPosEmbedding
from ddpm.modules.upsampling import Upsample

__all__ = [
    "UNet",
]


class UNet(Module):
    def __init__(
            self,
            dim: int,
            init_dim: Optional[int] = None,
            out_dim: Optional[int] = None,
            in_channels: int = 3,
            channels_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
            self_condition: bool = False,
            learned_variance: bool = False,
            sinusoidal_pos_emb_theta: int = 10000,
            dropout: float = 0.,
            attn_dim_head: int = 32,
            attn_heads: int = 4,
    ) -> None:
        """
        Initializes the UNet module.

        Args:
            dim (int): Base dimension for the model.
            init_dim (Optional[int], optional): Initial dimension for the first convolution. Defaults to None.
            out_dim (Optional[int], optional): Output dimension. Defaults to None.
            in_channels (int): Number of input channels. Defaults to 3.
            channels_multipliers (Tuple[int, ...]): Multipliers for the channels at each stage. Defaults to (1, 2, 4, 8).
            self_condition (bool): Whether to use self-conditioning. Defaults to False.
            learned_variance (bool): Whether to learn variance. Defaults to False.
            sinusoidal_pos_emb_theta (int): Theta value for sinusoidal positional embedding. Defaults to 10000.
            dropout (float): Dropout rate. Defaults to 0.
            attn_dim_head (int): Dimension of each attention head. Defaults to 32.
            attn_heads (int): Number of attention heads. Defaults to 4.
        """
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.self_condition = self_condition

        in_channels = in_channels * (2 if self_condition else 1)
        init_dim = dim if init_dim is None else init_dim
        self.init_conv = nn.Conv2d(in_channels, init_dim, (7, 7), padding=(3, 3))

        dim_list = [dim, *map(lambda m: dim * m, channels_multipliers)]
        out_channel_list = list(zip(dim_list[:-1], dim_list[1:]))

        # time embedding
        sinusoidal_pos_embedding = SinusoidalPosEmbedding(dim, sinusoidal_pos_emb_theta)
        fourier_dim = dim
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            sinusoidal_pos_embedding,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention
        num_stages = len(channels_multipliers)
        full_attn = (*((False,) * (num_stages - 1)), True)
        attn_heads = ((attn_heads,) * num_stages)
        attn_dim_head = ((attn_dim_head,) * num_stages)
        assert len(full_attn) == len(channels_multipliers)

        # prepare blocks
        attention_block = partial(Attention, flash=True)
        residual_block = partial(Residual, time_emb_dim=time_dim, dropout=dropout)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(out_channel_list)

        for i, (
                (in_dim, out_dim),
                layer_full_attn,
                layer_attn_heads,
                layer_attn_dim_head
        ) in enumerate(
            zip(
                out_channel_list,
                full_attn,
                attn_heads,
                attn_dim_head
            )
        ):
            is_last = i >= (num_resolutions - 1)
            attn_klass = attention_block if layer_full_attn else LinearAttention
            self.downs.append(nn.ModuleList([
                residual_block(in_dim, in_dim),
                residual_block(in_dim, in_dim),
                attn_klass(in_dim, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(in_dim, out_dim) if not is_last else nn.Conv2d(in_dim, out_dim, (3, 3), padding=(1, 1))
            ]))

        midden_dim = dim_list[-1]
        self.mid_block1 = residual_block(midden_dim, midden_dim)
        self.mid_attn = attention_block(midden_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = residual_block(midden_dim, midden_dim)

        for i, (
                (in_dim, out_dim),
                layer_full_attn,
                layer_attn_heads,
                layer_attn_dim_head
        ) in enumerate(zip(*map(reversed, (out_channel_list, full_attn, attn_heads, attn_dim_head)))):
            is_last = i == (len(out_channel_list) - 1)
            attn_klass = attention_block if layer_full_attn else LinearAttention
            self.ups.append(nn.ModuleList([
                residual_block(out_dim + in_dim, out_dim),
                residual_block(out_dim + in_dim, out_dim),
                attn_klass(out_dim, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(out_dim, in_dim) if not is_last else nn.Conv2d(out_dim, in_dim, (3, 3), padding=(1, 1))
            ]))

        default_out_dim = in_channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim if default_out_dim is None else default_out_dim

        self.final_res_block = residual_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward(self, x: torch.Tensor, sample_time: torch.Tensor, self_conditioning_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for the UNet module.

        Args:
            x (torch.Tensor): Input tensor.
            sample_time (torch.Tensor): Time embedding tensor.
            self_conditioning_tensor (Optional[torch.Tensor], optional): Self-conditioning tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after passing through the UNet.
        """
        # Ensure input dimensions are divisible by the downsample factor
        assert all(
            [
                (dimension % self.downsample_factor) == 0 for dimension in x.shape[-2:]
            ]
        ), f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"

        # Handle self-conditioning
        if self.self_condition:
            self_conditioning_tensor = torch.zeros_like(x) if self_conditioning_tensor is None else self_conditioning_tensor
            x = torch.cat((self_conditioning_tensor, x), dim=1)

        # Initial convolution
        x = self.init_conv(x)
        residual = x.clone()

        # Time embedding
        time_embedding = self.time_mlp(sample_time)

        hidden_states = []

        # Downsampling path
        for block1, block2, attention, downsample in self.downs:
            x = block1(x, time_embedding)
            hidden_states.append(x)
            x = block2(x, time_embedding)
            x = attention(x) + x
            hidden_states.append(x)
            x = downsample(x)

        # Middle block
        x = self.mid_block1(x, time_embedding)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time_embedding)

        # Upsampling path
        for block1, block2, attention, upsample in self.ups:
            x = torch.cat((x, hidden_states.pop()), 1)
            x = block1(x, time_embedding)
            x = torch.cat((x, hidden_states.pop()), 1)
            x = block2(x, time_embedding)
            x = attention(x) + x
            x = upsample(x)

        # Final block
        x = torch.cat((x, residual), 1)
        x = self.final_res_block(x, time_embedding)
        x = self.final_conv(x)

        return x

    @property
    def downsample_factor(self) -> int:
        """Returns the downsample factor of the UNet.

        Returns:
            int: Downsample factor.
        """
        return 2 ** (len(self.downs) - 1)
