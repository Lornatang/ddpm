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
from pathlib import Path
from typing import Dict, Union

import torch
import yaml

__all__ = [
    "extract", "num_to_groups", "save_yaml",
]


def extract(x: torch.Tensor, sample_time: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gather per-sample schedule values and reshape them for broadcasting.

    Args:
        x (torch.Tensor): One-dimensional schedule tensor.
        sample_time (torch.Tensor): Per-sample time-step indices.
        shape (torch.Size): Target tensor shape whose rank determines the broadcast dimensions.

    Returns:
        torch.Tensor: Gathered values with shape ``(batch, 1, ..., 1)``.
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
        file_path (Path | str, optional): Output YAML path. Defaults to ``"opts.yaml"``.
        data (Dict | None, optional): Data to save. Defaults to ``None``, which writes an empty dictionary.
    """
    if data is None:
        data = {}

    file_path = Path(file_path)

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    serializable = {k: v if isinstance(v, valid_types) else str(v) for k, v in data.items()}

    # Write data to file in YAML format
    with open(file_path, "w", errors="ignore", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, indent=4, allow_unicode=True, sort_keys=False)
