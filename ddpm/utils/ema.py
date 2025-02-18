# Copyright (c) AlphaBetter. All rights reserved.
from copy import deepcopy
from typing import Any, Tuple

import math
from torch import nn

from .torch_utils import de_parallel

__all__ = [
    "ModelEMA",
]


class ModelEMA:

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0) -> None:
        """Initialize the ModelEMA class.

        Args:
            model (nn.Module): The original model to maintain EMA.
            decay (float, optional): The decay rate for EMA, default is 0.9999.
            tau (int, optional): Time constant for calculating dynamic decay rate, default is 2000.
            updates (int, optional): Number of updates performed, default is 0.
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
            include (Tuple, optional): Names of attributes to include, default is empty.
            exclude (Tuple, optional): Names of attributes to exclude, default is empty.
        """
        # Iterate over all attributes of the new model
        for k, item in new_state_dict.__dict__.items():
            if (len(include) and k not in include) or k.startswith("_") or k in exclude:
                continue
            else:
                setattr(old_state_dict, k, item)

    def update(self, model: nn.Module) -> None:
        """Update the parameters of the EMA model using exponential moving average based on the current model's parameters and decay rate.

        Args:
            model (nn.Module): The current training model.
        """
        # Increment the number of updates
        if self.enabled:
            self.updates += 1

            # Calculate the current decay rate
            decay = self.decay(self.updates)

            # Get the state dictionary of the original model, ensuring to handle multi-GPU parallel models
            state_dict = de_parallel(model).state_dict()
            for key, value in self.ema.state_dict().items():
                if value.dtype.is_floating_point:
                    value *= decay
                    value += (1 - decay) * state_dict[key].detach()

    def update_attr(self, model: nn.Module, include: tuple = (), exclude: tuple = ("process_group", "reducer")) -> None:
        """Update the attributes of the EMA model to match specific attributes of the original model.

        Args:
            model (nn.Module): The original model.
            include (tuple, optional): Names of attributes to include.
            exclude (tuple, optional): Names of attributes to exclude, default is ("process_group", "reducer").
        """
        self.copy_attr(self.ema, model, include, exclude)
