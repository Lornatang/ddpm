# Copyright (c) Lornatang. All rights reserved.
from pathlib import Path
from typing import Union, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ddpm.data.base import BaseImageFolder
from ddpm.data.utils import cycle_data_loader
from ddpm.evaluation.fid import FIDMetrics
from ddpm.utils.events import LOGGER
from ddpm.utils.torch_utils import select_device


class Validator:
    def __init__(
            self,
            model_weights_path: Union[Path, str],
            root: Union[Path, str],
            batch_size: int = 16,
            num_samples: int = 16,
            num_fid_samples: int = 50000,
            fid_stat_path: Union[Path, str] = "./datasets/stats/train.npz",
            save_dir: Union[Path, str] = "./results/val",
            device: str = "0",
    ) -> None:
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
        checkpoint = torch.load(self.model_weights_path, weights_only=False)
        self.model = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]

        # AMP
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)

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
        self.model.eval()
        inception_score, fid_score = self.fid_metrics.calculate()
        LOGGER.info(f"Inception score: {inception_score:.2f}, FID score: {fid_score:.2f}")

        return inception_score, fid_score
