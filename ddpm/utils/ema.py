# Copyright (c) AlphaBetter. All rights reserved.
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
from copy import deepcopy
from collections import defaultdict
from typing import Any, Tuple

import math
import torch
from torch import nn

from .torch_utils import de_parallel

__all__ = [
    "ModelEMA",
]


class ModelEMA:

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0) -> None:
        """Initialize an evaluation-mode exponential moving average of the model.

        Args:
            model (nn.Module): Training model, optionally wrapped in a parallel module.
            decay (float, optional): Maximum EMA decay. Defaults to 0.9999.
            tau (int, optional): Time constant controlling decay warmup. Defaults to 2000.
            updates (int, optional): Number of previous EMA updates. Defaults to 0.
        """
        # Copy the model and set it to evaluation mode
        self.ema = deepcopy(de_parallel(model)).eval()

        # Initialize the number of updates
        self.updates = updates
        # Define the dynamic decay rate function, which gradually approaches the fixed decay rate as the number of updates increases
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))

        # Disable gradient computation for all parameters
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.enabled = True

    @staticmethod
    def copy_attr(old_state_dict: Any, new_state_dict: Any, include: Tuple = (), exclude: Tuple = ()) -> None:
        """Copy attributes from the new model to the old model.

        Args:
            old_state_dict (Any): Target object.
            new_state_dict (Any): Source model.
            include (Tuple, optional): Names to include; an empty tuple includes all public attributes. Defaults to ``()``.
            exclude (Tuple, optional): Names to exclude. Defaults to ``()``.
        """
        # Iterate over all attributes of the new model
        for k, item in new_state_dict.__dict__.items():
            if (len(include) and k not in include) or k.startswith("_") or k in exclude:
                continue
            else:
                setattr(old_state_dict, k, item)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update floating-point EMA parameters and buffers from the training model.

        Args:
            model (nn.Module): Current training model, optionally wrapped in a parallel module.

        Note:
            Non-floating buffers are left unchanged, and no update is performed when EMA is disabled.
        """
        # Increment the number of updates
        if self.enabled:
            self.updates += 1

            # Calculate the current decay rate
            decay = self.decay(self.updates)

            # Get the state dictionary of the original model, ensuring to handle multi-GPU parallel models
            state_dict = de_parallel(model).state_dict()
            groups = defaultdict(lambda: ([], []))
            for key, value in self.ema.state_dict().items():
                if value.dtype.is_floating_point:
                    source = state_dict[key].detach()
                    if value.layout != torch.strided or source.layout != torch.strided or value.dtype != source.dtype or value.device != source.device:
                        value.mul_(decay).add_(source * (1 - decay))
                    else:
                        targets, sources = groups[(value.device, value.dtype)]
                        targets.append(value)
                        sources.append(source)
            for targets, sources in groups.values():
                torch._foreach_mul_(targets, decay)
                torch._foreach_add_(targets, sources, alpha=1 - decay)

    def update_attr(self, model: nn.Module, include: tuple = (), exclude: tuple = ("process_group", "reducer")) -> None:
        """Update the attributes of the EMA model to match specific attributes of the original model.

        Args:
            model (nn.Module): The original model.
            include (tuple, optional): Names to include; an empty tuple includes all public attributes. Defaults to ``()``.
            exclude (tuple, optional): Names to exclude. Defaults to ``("process_group", "reducer")``.
        """
        self.copy_attr(self.ema, model, include, exclude)
