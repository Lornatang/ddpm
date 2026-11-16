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
"""Fréchet Inception Distance with matched real and generated preprocessing."""

import hashlib
import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from torch import distributed as dist, nn
from torch.nn import functional as F_torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm.utils.events import LOGGER
from ddpm.utils.ops import num_to_groups

__all__ = ["FIDMetrics"]


def frechet_distance(mean1, covariance1, mean2, covariance2, epsilon=1e-6) -> float:
    """Compute FID using the scipy sqrtm API shared by old and new releases."""
    mean1, mean2 = np.atleast_1d(mean1), np.atleast_1d(mean2)
    covariance1, covariance2 = np.atleast_2d(covariance1), np.atleast_2d(covariance2)
    if mean1.shape != mean2.shape or covariance1.shape != covariance2.shape:
        raise ValueError("FID feature statistics have incompatible shapes")
    root = linalg.sqrtm(covariance1 @ covariance2)
    if not np.isfinite(root).all():
        offset = np.eye(covariance1.shape[0]) * epsilon
        root = linalg.sqrtm((covariance1 + offset) @ (covariance2 + offset))
    if np.iscomplexobj(root):
        if not np.allclose(np.diag(root).imag, 0, atol=1e-3):
            raise ValueError("FID covariance square root has a large imaginary component")
        root = root.real
    difference = mean1 - mean2
    score = difference @ difference + np.trace(covariance1) + np.trace(covariance2) - 2 * np.trace(root)
    return float(max(score, 0.0))


def feature_statistics(features: np.ndarray, chunk_size: int = 8192):
    """Compute sample mean/covariance in float64 with bounded extra memory."""
    if features.ndim != 2 or len(features) < 2 or chunk_size < 1:
        raise ValueError("FID statistics require at least two feature vectors")
    count = 0
    mean = np.zeros(features.shape[1], dtype=np.float64)
    moments = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    for start in range(0, len(features), chunk_size):
        block = features[start:start + chunk_size].astype(np.float64, copy=True)
        block_mean = block.mean(axis=0)
        block -= block_mean
        block_count = len(block)
        total = count + block_count
        delta = block_mean - mean
        moments += block.T @ block
        if count:
            moments += np.outer(delta, delta) * (count * block_count / total)
        mean += delta * (block_count / total)
        count = total
    return mean, moments / (count - 1)


class FIDMetrics:
    def __init__(
        self,
        data_loader: DataLoader,
        batch_size: int,
        model: nn.Module,
        dataset_stat_path: Union[Path, str],
        device: torch.device = torch.device("cuda:0"),
        num_samples: int = 50000,
        feature_dims: int = 2048,
    ) -> None:
        if batch_size < 1 or num_samples < 2:
            raise ValueError("FID requires a positive batch size and at least two samples")
        if feature_dims not in InceptionV3.BLOCK_INDEX_BY_DIM:
            raise ValueError(f"Unsupported Inception feature dimension: {feature_dims}")
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.model = model
        self.dataset_stat_path = Path(dataset_stat_path)
        self.device = torch.device(device)
        self.num_samples = num_samples
        self.feature_dims = feature_dims
        dataset = getattr(data_loader, "dataset", None)
        if dataset is not None and len(dataset) < num_samples:
            raise ValueError(f"FID needs {num_samples} distinct real images, found {len(dataset)}")
        self.cache_signature = self._cache_signature(dataset)
        # Recreating the metric after resume must not change training RNG state.
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self.inception_v3 = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[feature_dims]]).to(self.device).eval()
        self.dataset_stat_loaded = False
        self.m2 = self.s2 = None

    def _cache_signature(self, dataset) -> str:
        metadata = {
            "format": 2, "feature_dims": self.feature_dims, "num_samples": self.num_samples,
            "dataset": type(dataset).__qualname__ if dataset is not None else None,
            "root": str(Path(dataset.root).resolve()) if hasattr(dataset, "root") else None,
            "length": len(dataset) if dataset is not None else None,
            "transform": repr(getattr(dataset, "transform", None)),
        }
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        # Invalidate cached statistics when images are added or modified.
        for path in getattr(dataset, "samples", ()):
            path = Path(path)
            stat = path.stat()
            digest.update(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        return digest.hexdigest()

    def calculate_inception_features(self, samples: torch.Tensor) -> torch.Tensor:
        """Extract features from RGB tensors in the range [0, 1]."""
        features = self.inception_v3(samples)[0]
        if features.shape[-2:] != (1, 1):
            features = F_torch.adaptive_avg_pool2d(features, (1, 1))
        return features.flatten(start_dim=1)

    def _cached_stats(self, path: Path):
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["cache_signature"].item()) == self.cache_signature:
                    return data["m2"], data["s2"]
        except (OSError, ValueError, KeyError):
            pass
        return None

    @torch.inference_mode()
    def load_dataset_stat(self) -> None:
        """Load compatible statistics or calculate them from exactly num_samples images."""
        path = self.dataset_stat_path
        stats = self._cached_stats(path)
        if path.exists() and stats is None:
            # Preserve old or incompatible files; don't mistake them for these statistics.
            path = path.with_name(f"{path.stem}-{self.cache_signature[:12]}{path.suffix or '.npz'}")
            stats = self._cached_stats(path)
        if stats is not None:
            self.m2, self.s2 = stats
            LOGGER.info(f"Dataset stats loaded from {path}")
        else:
            features = None
            count = 0
            for real_samples in tqdm(self.data_loader, desc="Real Inception features", total=len(self.data_loader)):
                real_samples = real_samples[: self.num_samples - count].to(self.device)
                batch = self.calculate_inception_features(real_samples).cpu().numpy()
                if features is None:
                    features = np.empty((self.num_samples, batch.shape[1]), dtype=np.float32)
                features[count:count + len(batch)] = batch
                count += len(real_samples)
                if count == self.num_samples:
                    break
            if count != self.num_samples:
                raise ValueError(f"FID expected {self.num_samples} real images, received {count}")
            self.m2, self.s2 = feature_statistics(features)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, m2=self.m2, s2=self.s2, cache_signature=self.cache_signature)
            LOGGER.info(f"Dataset stats cached to {path}")
        self.dataset_stat_loaded = True

    @torch.inference_mode()
    def calculate(self) -> float:
        if not self.dataset_stat_loaded:
            self.load_dataset_stat()
        self.model.eval()
        features = None
        count = 0
        for size in tqdm(num_to_groups(self.num_samples, self.batch_size), desc="Generated Inception features"):
            samples = self.model.sample(batch_size=size)
            batch = self.calculate_inception_features(samples).cpu().numpy()
            if features is None:
                features = np.empty((self.num_samples, batch.shape[1]), dtype=np.float32)
            features[count:count + size] = batch
            count += size
        mean, covariance = feature_statistics(features)
        return frechet_distance(mean, covariance, self.m2, self.s2)

    @torch.inference_mode()
    def calculate_distributed(self) -> float:
        """Generate exactly num_samples across ranks and compute FID on rank 0."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self.calculate()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if rank == 0 and not self.dataset_stat_loaded:
            self.load_dataset_stat()
        dist.barrier()
        self.model.eval()
        counts = [self.num_samples // world_size + (r < self.num_samples % world_size) for r in range(world_size)]
        local_batch = max(1, self.batch_size // world_size)
        rounds = (max(counts) + local_batch - 1) // local_batch
        features = np.empty((self.num_samples, self.feature_dims), dtype=np.float32) if rank == 0 else None
        count = 0
        for step in tqdm(range(rounds), desc="Distributed generated Inception features", disable=rank != 0):
            size = min(local_batch, max(0, counts[rank] - step * local_batch))
            outgoing = torch.zeros((local_batch, self.feature_dims), dtype=torch.float32, device=self.device)
            if size:
                batch = self.calculate_inception_features(self.model.sample(batch_size=size))
                if batch.shape != (size, self.feature_dims):
                    raise ValueError(f"Inception returned unexpected feature shape: {tuple(batch.shape)}")
                outgoing[:size] = batch
            gathered = [torch.empty_like(outgoing) for _ in range(world_size)]
            dist.all_gather(gathered, outgoing)
            if rank == 0:
                for r, received in enumerate(gathered):
                    valid = min(local_batch, max(0, counts[r] - step * local_batch))
                    if valid:
                        features[count:count + valid] = received[:valid].cpu().numpy()
                        count += valid
        score = torch.zeros((), dtype=torch.float64, device=self.device)
        if rank == 0:
            mean, covariance = feature_statistics(features)
            score.fill_(frechet_distance(mean, covariance, self.m2, self.s2))
        dist.broadcast(score, src=0)
        return float(score.item())
