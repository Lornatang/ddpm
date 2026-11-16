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
            batch_size: int = 256,
            num_samples: int = 100,
            save_dir: Union[Path, str] = "./results/inference",
            device: str = "0",
    ) -> None:
        """Initializes the Inferencer class for model inference.

        Args:
            model_weights_path (Union[Path, str]): Path to the pre-trained model weights.
            batch_size (int, optional): Batch size for inference. Defaults to 256.
            num_samples (int, optional): Number of samples to generate for evaluation. Defaults to 8.
            save_dir (Union[Path, str], optional): Directory to save the validation results. Defaults to './results/val'.
            device (str, optional): The device to use for computation. Defaults to '0' (CUDA).
        """
        if num_samples < 1 or batch_size < 1:
            raise ValueError("num_samples and batch_size must be positive")
        self.model_weights_path = model_weights_path
        self.batch_size = batch_size
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
            for i in tqdm(range(0, self.num_samples, self.batch_size), desc="Saving samples", unit="batch"):
                current_batch_size = min(self.batch_size, self.num_samples - i)
                samples = self.model.sample(batch_size=current_batch_size)
                for sample in samples:
                    vutils.save_image(sample, self.save_dir.joinpath(f"{uuid.uuid4()}.png"))

        end_time = time.time()
        total_time = end_time - start_time
        avg_sample_time = total_time / self.num_samples

        LOGGER.info(f"Generated {self.num_samples} samples in {total_time:.4f} seconds.")
        LOGGER.info(f"Average time per sample: {avg_sample_time:.4f} seconds.")
        LOGGER.info(f"Generated samples saved at {self.save_dir}")
