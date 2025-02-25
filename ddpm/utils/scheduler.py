# Copyright (c) Lornatang. All rights reserved.
from typing import Union

import math
import torch

__all__ = [
    "cosine_beta_schedule", "linear_beta_schedule", "sigmoid_beta_schedule",
]


def cosine_beta_schedule(time_steps: int, s: float = 0.008) -> torch.Tensor:
    """Generates a cosine beta schedule for the diffusion process.
    This schedule is proposed in 'Improved Denoising Diffusion Probabilistic Models'.

    Args:
        time_steps (int): The number of time steps in the diffusion process.
        s (float, optional): Offset parameter to prevent singularity at t=0. Defaults to 0.008.

    Returns:
        torch.Tensor: A tensor containing the cosine beta schedule values.
    """
    steps = time_steps + 1
    time_step = torch.linspace(0, time_steps, steps, dtype=torch.float64) / time_steps
    alphas_cumprod = torch.cos((time_step + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas.clip_(0, 0.999)

    return betas


def linear_beta_schedule(time_steps: int) -> torch.Tensor:
    """Generates a linear beta schedule for the diffusion process.

    Args:
        time_steps (int): The number of time steps in the diffusion process.

    Returns:
        torch.Tensor: A tensor containing the linear beta schedule values.
    """
    scale = 1000 / time_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(beta_start, beta_end, time_steps, dtype=torch.float64)

    return betas


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
    betas.clip_(0, 0.999)

    return betas
