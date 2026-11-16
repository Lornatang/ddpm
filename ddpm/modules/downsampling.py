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
from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import nn

__all__ = [
    "Downsample",
]


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        """Initializes the Downsample module.

        Args:
            in_channels (int): Dimension of the input.
            out_channels (Optional[int], optional): Dimension of the output. Defaults to None.
        """
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.downsample: nn.Module = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(in_channels * 4, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Downsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after downsampling and convolution.
        """
        return self.downsample(x)
