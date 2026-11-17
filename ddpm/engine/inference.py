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
from torch import distributed as dist
from torchvision import utils as vutils
from tqdm import tqdm

from ddpm.utils.checkpoint import load_checkpoint
from ddpm.utils.distributed import DistributedInference
from ddpm.utils.events import LOGGER

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
        """Configure image generation from a trusted checkpoint.

        Args:
            model_weights_path (Path | str): Path to a trusted checkpoint; EMA weights are preferred.
            batch_size (int, optional): Target global sampling batch size. Defaults to 256.
            num_samples (int, optional): Total number of images to generate across ranks. Defaults to 100.
            save_dir (Path | str, optional): Output directory for images. Defaults to ``"./results/inference"``.
            device (str, optional): Device specification. Defaults to ``"0"``; torchrun selects CUDA by local rank.

        Raises:
            ValueError: If the sample count or batch size is not positive.
        """
        if num_samples < 1 or batch_size < 1:
            raise ValueError("num_samples and batch_size must be positive")
        self.model_weights_path = model_weights_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.distributed = DistributedInference(device)
        self.device = self.distributed.device

        # Model
        try:
            self.model, _ = load_checkpoint(self.model_weights_path, self.device)
            self.model.float()
        except BaseException:
            self.distributed.close()
            raise

    def inference(self) -> None:
        """Generate the requested number of images and save them across ranks.

        The target batch is divided between ranks, with at least one sample per
        active rank batch. The slowest rank determines the reported duration.
        """
        group = self.distributed
        count = self.num_samples // group.world_size + (group.rank < self.num_samples % group.world_size)
        batch_size = max(1, self.batch_size // group.world_size)
        try:
            if group.world_size > 1:
                dist.barrier()
            start_time = time.perf_counter()
            with torch.inference_mode():
                for i in tqdm(range(0, count, batch_size), desc="Saving samples", unit="batch", disable=not group.is_main):
                    current_batch_size = min(batch_size, count - i)
                    samples = self.model.sample(batch_size=current_batch_size)
                    for sample in samples:
                        vutils.save_image(sample, self.save_dir.joinpath(f"{uuid.uuid4()}.png"))
            total_time = time.perf_counter() - start_time
            if group.world_size > 1:
                elapsed = torch.tensor(total_time, device=self.device, dtype=torch.float64)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                total_time = elapsed.item()
            if group.is_main:
                LOGGER.info(f"Generated {self.num_samples} samples in {total_time:.4f} seconds.")
                LOGGER.info(f"Average time per sample: {total_time / self.num_samples:.4f} seconds.")
                LOGGER.info(f"Generated samples saved at {self.save_dir}")
        finally:
            group.close()
