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
import argparse
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ddpm.utils.checkpoint import load_checkpoint
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
        default=None,
        help="Optional image size; must match the checkpoint"
    )
    parser.add_argument(
        "-n", "--num-time-steps",
        type=int,
        default=None,
        help="Optional number of time steps; must match the checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run computation on: 'cpu' or GPU number. Defaults to '0'"
    )

    return parser.parse_args()


@torch.inference_mode()
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
    if num_time_steps != model.num_time_steps:
        raise ValueError("num_time_steps must match the checkpoint schedule")
    result_list = []
    x_start = None
    for time_step in tqdm(range(num_time_steps - 1, -1, -1)):
        condition = x_start if model.self_condition else None
        x_t, x_start = model.p_sample(x_t, time_step, condition)
        if time_step % 100 == 0:
            result_list.append((x_t.permute(0, 2, 3, 1).cpu().numpy()[0] + 1) * 127.5)

    return result_list


def main() -> None:
    opts = get_opts()
    weights = Path(opts.weights)
    output_path = Path(opts.output_path)
    device = select_device(opts.device)

    model, _ = load_checkpoint(weights, device)
    image_size = tuple(model.image_size)
    if opts.image_size is not None and image_size != (opts.image_size, opts.image_size):
        raise ValueError("image_size must match the checkpoint")
    num_time_steps = model.num_time_steps if opts.num_time_steps is None else opts.num_time_steps

    x_t = torch.randn(1, model.in_channels, *image_size, dtype=torch.float32, device=device)

    result = reverse_diffusion(model, x_t, num_time_steps, device=device)
    result = np.concatenate(result, axis=1)
    result = np.clip(result, 0, 255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(result)).save(output_path)


if __name__ == "__main__":
    main()
