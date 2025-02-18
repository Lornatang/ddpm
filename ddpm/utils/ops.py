# Copyright (c) Lornatang. All rights reserved.
import torch


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
