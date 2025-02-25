# Copyright (c) Lornatang. All rights reserved.
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

__all__ = [
    "convert_optimizer_state_dict_to_half", "load_checkpoint",
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


def load_checkpoint(weights_path: Union[str, Path], device: Union[str, torch.device] = None) -> Tuple[Any, Any]:
    """Load the model checkpoint and perform post-processing.

    Args:
        weights_path (str, path): Path to the checkpoint file.
        device (str, torch.device): Device to use, default is None.
        fuse (bool): Whether to fuse model layers to optimize inference performance, default is False.

    Returns:
        Tuple (Any, Any): Returns the loaded model and the complete checkpoint dictionary.
    """
    # Load the checkpoint
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Get the model
    model = (checkpoint.get("ema") or checkpoint["model"]).to(device).float()
    # Record the checkpoint path
    model.eval()

    return model, checkpoint
