# Copyright (c) Lornatang. All rights reserved.
import torch

__all__ = [
    "extract", "num_to_groups",
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
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)

    return arr
