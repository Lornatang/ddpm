# Copyright (c) Lornatang. All rights reserved.
from pathlib import Path
from typing import Union

import math
import numpy as np
import torch
from einops import rearrange
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch import nn
from torch.nn import functional as F_torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ddpm.utils.events import LOGGER
from ddpm.utils.ops import num_to_groups

__all__ = [
    "FIDMetrics",
]

class FIDMetrics:
    def __init__(
            self,
            data_loader: DataLoader,
            batch_size: int,
            model: nn.Module,
            dataset_stat_path: Union[Path, str],
            device: torch.device = torch.device("cuda:0"),
            num_samples: int = 50000,
    ) -> None:
        """Initialize the FID class.

        Args:
            data_loader (DataLoader): DataLoader object for loading the dataset.
            batch_size (int): Number of samples per batch.
            model (nn.Module): Model used to generate samples.
            dataset_stat_path (Union[Path, str]): Directory to save or load dataset statistics.
            device (torch.device): Device to run the computations on (e.g., 'cuda:0' or 'cpu'). Defaults to ``torch.device("cuda:0")``.
            num_samples (int): Number of samples to use for FID calculation. Defaults to 50000.
        """
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.sampler = model
        self.dataset_stat_path = dataset_stat_path
        self.device = device
        self.num_samples = num_samples

        # Initialize InceptionV3 model for feature extraction
        self.inception_v3 = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        self.dataset_stat_loaded = False
        self.m2 = None  # Mean of real dataset features
        self.s2 = None  # Covariance of real dataset features

    def calculate_inception_features(self, samples: torch.Tensor) -> torch.Tensor:
        """Calculate Inception features for given samples.

        Args:
            samples (torch.Tensor): Input samples (images).

        Returns:
            torch.Tensor: Extracted features.
        """
        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = F_torch.adaptive_avg_pool2d(features, (1, 1))

        features = rearrange(features, "... 1 1 -> ...")

        return features

    def load_dataset_stat(self) -> None:
        """Load or calculate dataset statistics (mean and covariance of Inception features)."""
        try:
            ckpt = np.load(str(self.dataset_stat_path))
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            LOGGER.info(f"Dataset stats loaded from {self.dataset_stat_path}")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.num_samples / self.batch_size))
            stacked_real_features = []
            LOGGER.info(f"Stacking Inception features for {self.num_samples} samples from the real dataset.")
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples = next(self.data_loader)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = (torch.cat(stacked_real_features, dim=0).cpu().numpy())
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            Path(self.dataset_stat_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(self.dataset_stat_path), m2=m2, s2=s2)
            LOGGER.info(f"Dataset stats cached to {self.dataset_stat_path} for future use.")
            self.m2, self.s2 = m2, s2

        self.dataset_stat_loaded = True

    @torch.inference_mode()
    def fid_score(self) -> float:
        """Calculate the FID score between generated samples and the real dataset.

        Returns:
            float: FID score.
        """
        if not self.dataset_stat_loaded:
            self.load_dataset_stat()

        self.sampler.eval()
        batches = num_to_groups(self.num_samples, self.batch_size)
        stacked_fake_features = []
        LOGGER.info(f"Stacking Inception features for {self.num_samples} generated samples.")
        for batch in tqdm(batches):
            fake_samples = self.sampler.sample(batch_size=batch)
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)

        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)
