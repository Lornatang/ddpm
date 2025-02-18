# Copyright (c) Lornatang. All rights reserved.
import torch
from typing import Union

__all__ = [
    "sigmoid_beta_schedule",
]

def sigmoid_beta_schedule(
        time_steps: int,
        start: Union[int, float] = -3,
        end: Union[int, float] = 3,
        tau: Union[int, float] = 1,
) -> torch.Tensor:
    """Generates a beta schedule using a sigmoid function.

    Args:
        time_steps (int): The number of time steps.
        start (Union[int, float]): The start value for the sigmoid function. Defaults to -3.
        end (Union[int, float]): The end value for the sigmoid function. Defaults to 3.
        tau (Union[int, float]): The tau value for the sigmoid function. Defaults to 1.

    Returns:
        torch.Tensor: The beta schedule.
    """
    steps = time_steps + 1
    t = torch.linspace(0, time_steps, steps, dtype=torch.float64, device="cpu") / time_steps
    v_start = torch.sigmoid(torch.tensor(start / tau, dtype=torch.float64, device="cpu"))
    v_end = torch.sigmoid(torch.tensor(end / tau, dtype=torch.float64, device="cpu"))
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas.clamp_(0, 0.999)
    
    return betas
