# Copyright (c) Lornatang. All rights reserved.
from typing import Dict

import torch

__all__ = [
    "convert_optimizer_state_dict_to_half",
]


def convert_optimizer_state_dict_to_half(state_dict: Dict) -> Dict:
    """
    Convert float32 tensors in the optimizer state dictionary to float16 to reduce memory usage.

    Args:
        state_dict (Dict): The state dictionary of the optimizer.

    Returns:
        state_dict (Dict): The converted state dictionary with float16 tensors.
    """
    # Iterate through all states in the optimizer state dictionary
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict
