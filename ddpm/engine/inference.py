# Copyright (c) Lornatang. All rights reserved.
import time
import uuid
from pathlib import Path
from typing import Union

import torch
from torchvision import utils as vutils
from tqdm import tqdm

from ddpm.utils.checkpoint import load_checkpoint
from ddpm.utils.events import LOGGER
from ddpm.utils.torch_utils import select_device

__all__ = [
    "Inferencer",
]


class Inferencer:
    def __init__(
            self,
            model_weights_path: Union[Path, str],
            num_samples: int = 100,
            save_dir: Union[Path, str] = "./results/inference",
            device: str = "0",
    ) -> None:
        """Initializes the Inferencer class for model inference.

        Args:
            model_weights_path (Union[Path, str]): Path to the pre-trained model weights.
            num_samples (int, optional): Number of samples to generate for evaluation. Defaults to 8.
            save_dir (Union[Path, str], optional): Directory to save the validation results. Defaults to './results/val'.
            device (str, optional): The device to use for computation. Defaults to '0' (CUDA).
        """
        self.model_weights_path = model_weights_path
        self.num_samples = num_samples
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = select_device(device)

        # Model
        self.model, _ = load_checkpoint(self.model_weights_path, self.device)
        self.model.float()

    def inference(self) -> None:
        """Perform model inference and save samples."""
        start_time = time.time()

        # Save samples
        with torch.no_grad():
            for _ in tqdm(range(self.num_samples), desc="Saving samples", unit="sample"):
                sample = self.model.sample(batch_size=1)
                vutils.save_image(sample, self.save_dir.joinpath(f"{uuid.uuid4()}.png"))

        end_time = time.time()
        total_time = end_time - start_time
        avg_sample_time = total_time / self.num_samples

        LOGGER.info(f"Generated {self.num_samples} samples in {total_time:.4f} seconds.")
        LOGGER.info(f"Average time per sample: {avg_sample_time:.4f} seconds.")
        LOGGER.info(f"Generated samples saved at {self.save_dir}")
