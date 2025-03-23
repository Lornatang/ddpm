# Copyright (c) Lornatang. All rights reserved.
import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F_torch
from tqdm import tqdm

from ddpm.models.unet import UNet
from ddpm.utils.scheduler import linear_beta_schedule
from ddpm.utils.torch_utils import select_device


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="./results/train/cifar10/weights/last.pkl",
        help="Path to the model weights path to be diffused. Defaults to './results/train/cifar10/weights/last.pkl'"
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        default="./results/analysis/reverse_diffusion_x_0.png",
        help="Path where the output visualization will be saved. Defaults to 'results/analysis/reverse_diffusion_x_0.png'"
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
        "--device",
        type=str,
        default="0",
        help="Device to run computation on: 'cpu' or GPU number. Defaults to '0'"
    )

    return parser.parse_args()


def reverse_diffusion(
        model: torch.nn.Module,
        x_t: torch.Tensor,
        num_time_steps: int,
        device: Union[str, torch.device],
) -> List[np.ndarray]:
    """Perform reverse diffusion process to generate images from noise.

    Args:
        model (torch.nn.Module): The trained diffusion model used for denoising.
        x_t (torch.Tensor): Input noise tensor of shape [B, C, H, W].
        num_time_steps (int): Total number of diffusion time steps.
        device (Union[str, torch.device]): Device to run computation on.

    Returns:
        List[np.ndarray]: List of intermediate denoised images at every 100 time steps,  with values normalized to [0, 255].
    """
    betas = linear_beta_schedule(num_time_steps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F_torch.pad(alphas_cumprod[:-1], pad=(1, 0), value=1.)
    result_list = []
    for time_step in tqdm(range(num_time_steps - 1, -1, -1)):
        time_step = torch.tensor([time_step], device=device)
        alphas_bar_t_1 = alphas_cumprod_prev.gather(-1, time_step).view(-1, 1, 1, 1)
        alphas_bar_t = alphas_cumprod.gather(-1, time_step).view(-1, 1, 1, 1)
        beta_t = betas.gather(-1, time_step).view(-1, 1, 1, 1)
        alpha_t = 1 - beta_t
        sigma = torch.sqrt((1 - alphas_bar_t_1) / (1 - alphas_bar_t) * beta_t)
        z = torch.randn_like(x_t, device=device)
        with torch.no_grad():
            mu_pred = (1 - alpha_t) / torch.sqrt(1 - alphas_bar_t) * model(x_t.float(), time_step.float())
        if time_step.cpu().numpy() > 0:
            x_t_1 = 1 / torch.sqrt(alpha_t) * (x_t - mu_pred) + sigma * z
        else:
            x_t_1 = 1 / torch.sqrt(alpha_t) * (x_t - mu_pred)
        x_t = x_t_1
        if time_step.cpu().numpy() % 100 == 0:
            result_list.append((x_t.permute(0, 2, 3, 1).cpu().numpy()[0] + 1) * 127.5)

    return result_list


def main() -> None:
    opts = get_opts()
    weights = Path(opts.weights)
    output_path = Path(opts.output_path)
    image_size = opts.image_size
    num_time_steps = opts.num_time_steps
    device = select_device(opts.device)

    model = UNet(128, channels_multipliers=(1, 2, 2, 2), dropout=0.1).to(device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema"].model.state_dict())

    x_t = torch.randn(1, 3, image_size, image_size, dtype=torch.float32, device=device)

    result = reverse_diffusion(model, x_t, num_time_steps, device=device)
    result = np.concatenate(result, axis=1)
    result = np.clip(result, 0, 255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(result)).save(output_path)


if __name__ == "__main__":
    main()
