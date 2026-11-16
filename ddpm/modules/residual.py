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
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn

from .base import Block

__all__ = [
    "Residual",
]


class Residual(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_embedding_dim: Optional[int] = None, dropout: float = 0.) -> None:
        """Initializes the Residual module.

        Args:
            dim (int): Dimension of the input.
            dim_out (int): Dimension of the output.
            time_embedding_dim (Optional[int], optional): Dimension of the time embedding. Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.
        """
        super().__init__()
        self.mlp: Optional[nn.Sequential] = None
        if time_embedding_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embedding_dim, dim_out * 2)
            )

        self.block1: Block = Block(dim, dim_out, dropout=dropout)
        self.block2: Block = Block(dim_out, dim_out)
        self.res_conv: Union[nn.Conv2d, nn.Identity] = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for the Residual module.

        Args:
            x (torch.Tensor): Input tensor.
            time_embedding (Optional[torch.Tensor], optional): Time embedding tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying the residual block operations.
        """
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.mlp is not None and time_embedding is not None:
            time_embedding = rearrange(time_embedding, "b c -> b c")
            time_embedding = self.mlp(time_embedding)
            time_embedding = rearrange(time_embedding, "b c -> b c 1 1")
            scale_shift = time_embedding.chunk(2, dim=1)

        h: torch.Tensor = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)
