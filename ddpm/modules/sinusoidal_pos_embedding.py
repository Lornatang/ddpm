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
