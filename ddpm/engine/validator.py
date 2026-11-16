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
import uuid
from pathlib import Path
from typing import Union

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from ddpm.data.base import BaseImageFolder, image_transform
from ddpm.evaluation.fid_metrics import FIDMetrics
from ddpm.utils.checkpoint import load_checkpoint
from ddpm.utils.distributed import DistributedInference
from ddpm.utils.events import LOGGER

__all__ = [
    "Validator",
]


class Validator:
    def __init__(
            self,
            model_weights_path: Union[Path, str],
            root: Union[Path, str],
            batch_size: int = 256,
            num_fid_samples: int = 50000,
            fid_stat_path: Union[Path, str] = "./datasets/stats/val.npz",
            save_dir: Union[Path, str] = "./results/val",
            device: str = "0",
            num_workers: int = 8,
            feature_dims: int = 2048,
    ) -> None:
        """Initializes the Validator class for model evaluation.

        Args:
            model_weights_path (Union[Path, str]): Path to the pre-trained model weights.
            root (Union[Path, str]): Root directory containing the dataset.
            batch_size (int, optional): Batch size for validation. Defaults to 256.
            num_fid_samples (int, optional): Number of samples to use for FID calculation. Defaults to 50000.
            fid_stat_path (Union[Path, str], optional): Path to the FID statistics. Defaults to './datasets/stats/train.npz'.
            save_dir (Union[Path, str], optional): Directory to save the validation results. Defaults to './results/val'.
            device (str, optional): The device to use for computation. Defaults to '0' (CUDA).
        """
        self.model_weights_path = model_weights_path
        self.root = root
        self.batch_size = batch_size
        self.num_fid_samples = num_fid_samples
        self.fid_stat_path = fid_stat_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.distributed = DistributedInference(device)
        self.device = self.distributed.device

        # Model
        try:
            self.model, _ = load_checkpoint(self.model_weights_path, self.device)
            self.model.float()
            transform = image_transform(self.model.image_size)
            self.dataset = BaseImageFolder(root, transform) if self.distributed.is_main else None
            self.data_loader = (DataLoader(self.dataset, batch_size, False, num_workers=num_workers,
                                           pin_memory=self.device.type == "cuda", persistent_workers=num_workers > 0)
                                if self.dataset is not None else None)
        except BaseException:
            self.distributed.close()
            raise

        # Metric computation
        self.feature_dims = feature_dims
        self.fid_metrics = None

    def validate(self) -> float:
        """Perform model evaluation and calculate FID score.

        Returns:
            float: The FID score of the model on the validation dataset.
        """
        group = self.distributed
        try:
            # Rank 0 finishes the first Inception download before other ranks load it.
            if group.is_main:
                self.fid_metrics = self._make_fid()
            if group.world_size > 1:
                dist.barrier()
            if not group.is_main:
                self.fid_metrics = self._make_fid()
            if group.is_main:
                LOGGER.info(f"Saving {self.batch_size} samples...")
            preview_count = self.batch_size // group.world_size + (group.rank < self.batch_size % group.world_size)
            if preview_count:
                with torch.inference_mode():
                    samples = self.model.sample(batch_size=preview_count)
                    for sample in samples:
                        vutils.save_image(sample, self.save_dir.joinpath(f"{uuid.uuid4()}.png"))
            fid_score = (self.fid_metrics.calculate_distributed() if group.world_size > 1
                         else self.fid_metrics.calculate())
            if group.is_main:
                LOGGER.info(f"FID score: {fid_score:.2f}")
            return fid_score
        finally:
            group.close()

    def _make_fid(self):
        return FIDMetrics(self.data_loader, self.batch_size, self.model,
                          dataset_stat_path=self.fid_stat_path, num_samples=self.num_fid_samples,
                          device=self.device, feature_dims=self.feature_dims)
