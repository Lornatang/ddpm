# Copyright (c) Lornatang. All rights reserved.
import uuid
from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm

from ddpm.data.base import BaseImageFolder
from ddpm.data.utils import cycle_data_loader
from ddpm.evaluation.fid import FIDMetrics
from ddpm.utils.checkpoint import load_checkpoint
from ddpm.utils.events import LOGGER
from ddpm.utils.torch_utils import select_device

__all__ = [
    "Validator",
]


class Validator:
    def __init__(
            self,
            model_weights_path: Union[Path, str],
            root: Union[Path, str],
            batch_size: int = 128,
            num_samples: int = 8,
            num_fid_samples: int = 50000,
            fid_stat_path: Union[Path, str] = "./datasets/stats/val.npz",
            save_dir: Union[Path, str] = "./results/val",
            device: str = "0",
    ) -> None:
        """Initializes the Validator class for model evaluation.

        Args:
            model_weights_path (Union[Path, str]): Path to the pre-trained model weights.
            root (Union[Path, str]): Root directory containing the dataset.
            batch_size (int, optional): Batch size for validation. Defaults to 128.
            num_samples (int, optional): Number of samples to generate for evaluation. Defaults to 8.
            num_fid_samples (int, optional): Number of samples to use for FID calculation. Defaults to 50000.
            fid_stat_path (Union[Path, str], optional): Path to the FID statistics. Defaults to './datasets/stats/train.npz'.
            save_dir (Union[Path, str], optional): Directory to save the validation results. Defaults to './results/val'.
            device (str, optional): The device to use for computation. Defaults to '0' (CUDA).
        """
        self.model_weights_path = model_weights_path
        self.root = root
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_fid_samples = num_fid_samples
        self.fid_stat_path = fid_stat_path
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = select_device(device, self.batch_size)

        # Model
        self.model, _ = load_checkpoint(self.model_weights_path, self.device)
        self.model.float()

        # Dataset and dataloader
        transform = transforms.Compose([
            transforms.Resize(self.model.image_size),
            transforms.CenterCrop(self.model.image_size),
            transforms.ToTensor()
        ])
        self.dataset = BaseImageFolder(root, transform)
        data_loader = DataLoader(self.dataset, batch_size, False, num_workers=8, pin_memory=True, persistent_workers=True)
        self.data_loader = cycle_data_loader(data_loader)

        # Metric computation
        self.fid_metrics = FIDMetrics(
            self.data_loader,
            self.batch_size,
            self.model,
            dataset_stat_path=self.fid_stat_path,
            num_samples=self.num_fid_samples,
            device=self.device,
        )

    def validate(self) -> Tuple[float, float]:
        """Perform model evaluation and calculate Inception score and FID score.

        Returns:
            Tuple[float, float]: The Inception score and FID score of the model on the validation dataset.
        """
        # Save samples
        with torch.no_grad():
            for _ in tqdm(range(self.num_samples), desc="Saving samples", unit="sample"):
                sample = self.model.sample(batch_size=1)
                vutils.save_image(sample, self.save_dir.joinpath(f"{uuid.uuid4()}.png"))

        # Calculate metrics
        inception_score, fid_score = self.fid_metrics.calculate()
        LOGGER.info(f"Inception score: {inception_score:.2f}, FID score: {fid_score:.2f}")

        return inception_score, fid_score
