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
import torch
from torch import nn
from torch.nn import functional as F_torch

__all__ = [
    "Attend",
]


class Attend(nn.Module):
    def __init__(
            self,
            dropout: float = 0.,
            flash: bool = False,
            scale: float = None
    ) -> None:
        """Initializes the Attend module.

        Args:
            dropout (float, optional): Dropout rate. Defaults to 0.
            flash (bool, optional): Whether to use flash attention. Defaults to False.
            scale (float, optional): Scaling factor for attention. Defaults to None.
        """
        super().__init__()
        self.dropout = dropout
        self.flash = flash
        self.scale = scale

        self.attn_dropout = nn.Dropout(dropout)

    def flash_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Perform scaled dot-product attention with automatic kernel selection.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, heads, seq_len_q, dim_head]
            k (torch.Tensor): Key tensor of shape [batch_size, heads, seq_len_k, dim_head]
            v (torch.Tensor): Value tensor of shape [batch_size, heads, seq_len_k, dim_head]

        Returns:
            torch.Tensor: Output attention tensor of shape [batch_size, heads, seq_len_q, dim_head]
        """
        # Let PyTorch choose Flash, memory-efficient or math attention for the
        # current hardware, dtype and inputs. Do not silently change the dtype.
        return F_torch.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0., scale=self.scale
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Attend module.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        if self.flash:
            return self.flash_attn(q, k, v)

        scale = q.shape[-1] ** -0.5 if self.scale is None else self.scale

        # similarity
        sim = torch.einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
