# Copyright (c) Lornatang. All rights reserved.
import argparse
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image

from ddpm.utils.scheduler import linear_beta_schedule
from ddpm.utils.torch_utils import select_device


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-path",
        type=str,
        default="./assets/x_0.png",
        help="Path to the input image file to be diffused. Defaults to './assets/x_0.png'"
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        default="./results/analysis/diffusion_x_0.png",
        help="Path where the output visualization will be saved. Defaults to './results/analysis/diffusion_x_0.png'"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Size to resize the input image (both width and height). Defaults to 32"
    )
    parser.add_argument(
        "-n", "--num-time-steps",
        type=int,
        default=1000,
        help="Total number of diffusion time steps in the process. Defaults to 1000"
    )
    parser.add_argument(
        "-t", "--time-steps",
        type=str,
        default="100,200,300,400,500,600,700,800,999",
        help="Comma-separated time steps at which to visualize the diffusion process (e.g., '100,200,300')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run computation on: 'cpu' or GPU number. Defaults to '0'"
    )

    return parser.parse_args()


def preprocess_image(
        image_path: Union[Path, str],
        target_size: Tuple[int, int] = None,
        normalize: bool = True,
) -> np.ndarray:
    """Preprocess an image for the diffusion model by loading, resizing and normalizing it.

    Args:
        image_path (Union[Path, str]): Path to the input image file.
        target_size (tuple[int, int], optional): Target size (width, height) to resize the image. Defaults to None.
        normalize (bool, optional): Whether to normalize the image to [-1, 1]. Defaults to True.

    Returns:
        np.ndarray: Preprocessed image tensor normalized to [-1, 1] with shape [C, H, W].
    """
    image = Image.open(image_path).convert("RGB")

    if target_size is not None:
        image = image.resize(target_size)

    image = np.array(image, dtype=np.float32).astype(np.float32) / 255.0

    if normalize:
        image = (image * 2.0) - 1.0

    return image


def diffusion(
        x_0: np.ndarray,
        num_time_steps: int,
        time_step: int,
        device: torch.device,
) -> torch.Tensor:
    """Apply the forward diffusion process to an input image.

    Args:
        x_0 (np.ndarray): The input image tensor to be diffused, normalized to [-1, 1].
        num_time_steps (int): Total number of diffusion time steps.
        time_step (int): Current diffusion time step.
        device (torch.device): Device to run the computation on.

    Returns:
        torch.Tensor: The diffused image tensor at time step t.
    """
    x_0 = torch.tensor(x_0).to(device)
    betas = linear_beta_schedule(num_time_steps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    alpha_bar = alphas_cumprod.gather(-1, torch.tensor(time_step, device=device))
    eps = torch.randn_like(x_0, device=device)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps

    return x_t


def main() -> None:
    opts = get_opts()
    image_path = Path(opts.image_path)
    output_path = Path(opts.output_path)
    image_size = (opts.image_size, opts.image_size)
    num_time_steps = opts.num_time_steps
    time_steps = opts.time_steps
    device = select_device(opts.device)

    x_0 = preprocess_image(image_path, image_size, True)

    noise_list = [(x_0 + 1.0) * 127.5]
    for time_step in time_steps.split(","):
        noise = diffusion(x_0, num_time_steps, int(time_step), device)
        noise = np.clip((noise.cpu().numpy() + 1.0) * 127.5, 0, 255)
        noise_list.append(noise)

    noise_list = np.concatenate(noise_list, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(noise_list)).save(output_path)


if __name__ == "__main__":
    main()
