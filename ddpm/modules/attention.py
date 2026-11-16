# Copyright (c) Lornatang. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from functools import partial

import torch
from einops import rearrange, repeat
from torch import nn

from .activation import RMSNorm
from .attend import Attend

__all__ = [
    "Attention", "LinearAttention",
]


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int = 4,
            dim_head: int = 32,
            num_mem_kv: int = 4,
            flash: bool = False
    ) -> None:
        """Initializes the Attention module.

        Args:
            dim (int): Dimension of the input.
            heads (int, optional): Number of attention heads. Defaults to 4.
            dim_head (int, optional): Dimension of each attention head. Defaults to 32.
            num_mem_kv (int, optional): Number of memory key-value pairs. Defaults to 4.
            flash (bool, optional): Whether to use flash attention. Defaults to False.
        """
        super().__init__()
        self.heads: int = heads
        hidden_dim = dim_head * heads

        self.norm: nn.Module = RMSNorm(dim)
        self.attend: Attend = Attend(flash=flash)

        self.mem_kv: nn.Parameter = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv: nn.Module = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out: nn.Module = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        if self.attend.flash:
            # SDPA's fused kernels require a contiguous final dimension and
            # matching dtypes, including the learned memory tokens under AMP.
            q = q.contiguous()
            mk, mv = mk.to(k.dtype), mv.to(v.dtype)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        out = self.to_out(out)

        return out


class LinearAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int = 4,
            dim_head: int = 32,
            num_mem_kv: int = 4
    ) -> None:
        """Initializes the LinearAttention module.

        Args:
            dim (int): Dimension of the input.
            heads (int, optional): Number of attention heads. Defaults to 4.
            dim_head (int, optional): Dimension of each attention head. Defaults to 32.
            num_mem_kv (int, optional): Number of memory key-value pairs. Defaults to 4.
        """
        super().__init__()
        self.scale: float = dim_head ** -0.5
        self.heads: int = heads
        hidden_dim = dim_head * heads

        self.norm: nn.Module = RMSNorm(dim)

        self.mem_kv: nn.Parameter = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv: nn.Module = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out: nn.Module = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            RMSNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the LinearAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying linear attention.
        """
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        out = self.to_out(out)

        return out
