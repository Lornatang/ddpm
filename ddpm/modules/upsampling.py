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
from torch import nn

__all__ = [
    "Upsample",
]


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        """Initializes the Upsample module.

        Args:
            in_channels (int): Dimension of the input.
            out_channels (Optional[int], optional): Dimension of the output. Defaults to None.
        """
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.upsample: nn.Module = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Upsample module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after upsampling and convolution.
        """
        return self.upsample(x)
