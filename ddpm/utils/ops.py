# Copyright (c) Lornatang. All rights reserved.
from pathlib import Path
from typing import Dict, Union

import torch
import yaml

__all__ = [
    "extract", "num_to_groups", "save_yaml",
]


def extract(x: torch.Tensor, sample_time: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Extracts values from a tensor `x` at indices specified by `sample_time` and reshapes the result.

    Args:
        x (torch.Tensor): The input tensor from which values are extracted.
        sample_time (torch.Tensor): A tensor containing the indices at which to extract values from `x`.
        shape (torch.Size): The desired shape of the output tensor.

    Returns:
        torch.Tensor: The extracted and reshaped tensor.
    """
    b, *_ = sample_time.shape
    out = x.gather(-1, sample_time)
    return out.reshape(b, *((1,) * (len(shape) - 1)))


def num_to_groups(num: int, divisor: int) -> list[int]:
    """Divide a number into groups of a specified size.

    Args:
        num (int): The total number to be divided.
        divisor (int): The size of each group.

    Returns:
        list[int]: A list containing the sizes of each group.
    """
    groups = num // divisor
    remainder = num % divisor
    array = [divisor] * groups
    if remainder > 0:
        array.append(remainder)

    return array


def save_yaml(file_path: Union[Path, str] = "opts.yaml", data: Dict = None) -> None:
    """Save dictionary data to a YAML file.

    Args:
        file_path (Union[Path, str]): Path where the YAML file will be saved. Defaults to "opts.yaml".
        data (Dict, optional): Dictionary data to be saved. Defaults to None (empty dict).
    """
    if data is None:
        data = {}

    file_path = Path(file_path)

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Write data to file in YAML format
    with open(file_path, "w", errors="ignore", encoding="utf-8") as f:
        yaml.safe_dump(data, f, indent=4, allow_unicode=True, sort_keys=False)
