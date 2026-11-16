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
"""Single-node or torchrun-launched distributed diffusion training."""

import math
import hashlib
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch import distributed as dist, nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import utils as vutils
from tqdm.auto import tqdm

from ddpm.data.base import BaseImageFolder, image_transform
from ddpm.evaluation.fid_metrics import FIDMetrics
from ddpm.utils.checkpoint import load_checkpoint, model_config
from ddpm.utils.ema import ModelEMA
from ddpm.utils.envs import set_seed_everything
from ddpm.utils.events import LOGGER
from ddpm.utils.ops import num_to_groups, save_yaml
from ddpm.utils.torch_utils import de_parallel


class Trainer:
    """Train a diffusion model. ``batch_size`` is the global microbatch size.

    Launch multiple GPU workers with torchrun. CPU/gloo is supported for tests.
    The effective optimizer batch is batch_size * gradient_accumulate_every.
    """

    def __init__(
        self,
        model: nn.Module,
        root: Union[Path, str],
        augment: bool = False,
        resume: Union[Path, str, None] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        gradient_accumulate_every: int = 1,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        num_steps: int = 100000,
        max_grad_norm: float = 1.0,
        val: bool = False,
        num_samples: int = 64,
        num_fid_samples: int = 50000,
        fid_stat_path: Union[Path, str] = "./datasets/stats/train.npz",
        save_and_sample_every: int = 1000,
        save_dir: Union[Path, str] = "./results/train",
        device: str = "cuda",
        seed: int = 0,
        fid_feature_dims: int = 2048,
    ) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_main = self.rank == 0
        if self.world_size < 1 or self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("Invalid torchrun RANK or WORLD_SIZE")
        if batch_size < 1 or batch_size % self.world_size:
            raise ValueError("batch_size must be positive and divisible by WORLD_SIZE")
        if num_workers < 0 or gradient_accumulate_every < 1 or num_steps < 1:
            raise ValueError("num_workers must be non-negative and training step counts must be positive")
        if save_and_sample_every < 1 or num_samples < 1 or max_grad_norm <= 0:
            raise ValueError("sampling interval, num_samples and max_grad_norm must be positive")

        self.root = str(root)
        self.augment = augment
        self.resume = str(resume) if resume else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gradient_accumulate_every = gradient_accumulate_every
        self.lr, self.betas = lr, betas
        self.num_steps, self.max_grad_norm = num_steps, max_grad_norm
        self.val, self.num_samples = val, num_samples
        self.num_fid_samples, self.fid_stat_path = num_fid_samples, str(fid_stat_path)
        self.fid_feature_dims = fid_feature_dims
        self.save_and_sample_every, self.seed = save_and_sample_every, seed
        self.device_type = str(device)
        self.device = self._select_device(device)
        self.save_dir = Path(self.resume).parent.parent if self.resume else Path(save_dir)
        self.save_samples_dir = self.save_dir / "samples"
        self.save_weights_dir = self.save_dir / "weights"
        if self.is_main:
            self.save_samples_dir.mkdir(parents=True, exist_ok=True)
            self.save_weights_dir.mkdir(parents=True, exist_ok=True)
            if not self.resume:
                save_yaml(self.save_dir / "opts.yaml", vars(self))

        set_seed_everything(seed + self.rank)
        self.model = model
        self.image_size = model.image_size
        self.current_step = 0
        self.data_epoch = 0
        self.batches_in_epoch = 0
        self._loader_iter = None
        self.ema = None
        self.optimizer = None
        self.scaler = None
        self.fid_metrics = None
        self.fid_score = None
        self.best_fid_score = float("inf")
        self.last_saved_step = -1

    def _select_device(self, device: str) -> torch.device:
        device = str(device).lower()
        if device == "cpu":
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a CUDA PyTorch build or use device='cpu' for testing")
        if self.world_size > 1:
            index = self.local_rank
        elif device in ("cuda", "auto", ""):
            index = 0
        elif device.startswith("cuda:"):
            index = int(device.split(":", 1)[1])
        else:
            index = int(device)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable")
        torch.cuda.set_device(index)
        return torch.device("cuda", index)

    def _setup_ddp(self) -> None:
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if self.device.type == "cuda" else "gloo",
                init_method="env://", timeout=timedelta(hours=12),
            )

    def _setup_train(self) -> None:
        checkpoint = None
        if self.resume:
            self.model, checkpoint = load_checkpoint(self.resume, self.device, model=self.model, prefer_ema=False)
        self.model = self.model.to(self.device)
        if self.world_size > 1:
            device_ids = [self.local_rank] if self.device.type == "cuda" else None
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=device_ids)

        dataset = BaseImageFolder(self.root, image_transform(self.image_size, augment=self.augment, normalize=True))
        if len(dataset) < self.batch_size:
            raise ValueError(f"Training needs at least batch_size={self.batch_size} images, found {len(dataset)}")
        names = "\0".join(str(path.relative_to(dataset.root)) for path in dataset.samples)
        self.resume_config = {
            "root": str(Path(self.root).resolve()), "dataset_size": len(dataset),
            "dataset_names_sha256": hashlib.sha256(names.encode("utf-8")).hexdigest(),
            "world_size": self.world_size, "device_type": self.device.type,
            "batch_size": self.batch_size, "num_workers": self.num_workers,
            "gradient_accumulate_every": self.gradient_accumulate_every,
            "max_grad_norm": self.max_grad_norm, "augment": self.augment, "seed": self.seed,
        }
        self.train_sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, seed=self.seed, drop_last=True,
        )
        self.data_generator = torch.Generator()
        self.train_loader = DataLoader(
            dataset, batch_size=self.batch_size // self.world_size, sampler=self.train_sampler,
            num_workers=self.num_workers, pin_memory=self.device.type == "cuda",
            persistent_workers=False, drop_last=True, generator=self.data_generator,
        )
        if not len(self.train_loader):
            raise ValueError("Training loader contains no complete batches")

        if self.is_main:
            self.ema = ModelEMA(self.model)
            if self.val:
                real_dataset = BaseImageFolder(self.root, image_transform(self.image_size))
                real_loader = DataLoader(
                    real_dataset, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.num_workers, pin_memory=self.device.type == "cuda",
                    persistent_workers=self.num_workers > 0,
                )
                self.fid_metrics = FIDMetrics(
                    real_loader, self.batch_size, self.ema.ema,
                    dataset_stat_path=self.fid_stat_path, num_samples=self.num_fid_samples,
                    device=self.device, feature_dims=self.fid_feature_dims,
                )

        self.optimizer = optim.Adam(self.model.parameters(), self.lr, self.betas)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        if checkpoint is not None:
            self.resume_training(checkpoint)

    def _rng_state(self) -> dict:
        return {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }

    def _set_rng_state(self, state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if self.device.type == "cuda" and state.get("cuda") is not None:
            torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)

    def resume_training(self, checkpoint: dict) -> None:
        previous_config = checkpoint.get("resume_config")
        if previous_config is not None:
            mismatches = [key for key, value in self.resume_config.items() if previous_config.get(key) != value]
            if mismatches:
                raise ValueError("Resume configuration differs from the checkpoint: " + ", ".join(mismatches))
        self.current_step = int(checkpoint.get("step", 0))
        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_fid_score = float(checkpoint.get("best_fid_score", float("inf")))
        if self.ema and checkpoint.get("ema") is not None:
            ema_state = checkpoint["ema"]
            if isinstance(ema_state, nn.Module):
                ema_state = ema_state.float().state_dict()
            self.ema.ema.load_state_dict(ema_state)
            self.ema.updates = int(checkpoint.get("updates", 0))
        self.data_epoch = int(checkpoint.get("data_epoch", 0))
        self.batches_in_epoch = int(checkpoint.get("batches_in_epoch", 0))
        if self.batches_in_epoch >= len(self.train_loader):
            raise ValueError("Checkpoint data position is incompatible with this dataset or batch size")
        rng_states = checkpoint.get("rng_states")
        if rng_states is not None:
            if len(rng_states) != self.world_size:
                raise ValueError("Resume requires the same WORLD_SIZE as the saved checkpoint")
            self._set_rng_state(rng_states[self.rank])
        else:
            warnings.warn("Legacy checkpoint lacks full training and RNG state; resumption is not exact", stacklevel=2)
        LOGGER.info(f"Resuming at step {self.current_step} of {self.num_steps}")

    def _next_batch(self):
        if self._loader_iter is None:
            self.train_sampler.set_epoch(self.data_epoch)
            self.data_generator.manual_seed(self.seed + self.data_epoch * self.world_size + self.rank)
            self._loader_iter = iter(self.train_loader)
            # Rebuild the same worker/sampler position without consuming model RNG.
            if self.batches_in_epoch:
                state = self._rng_state()
                for _ in range(self.batches_in_epoch):
                    next(self._loader_iter)
                self._set_rng_state(state)
        batch = next(self._loader_iter)
        self.batches_in_epoch += 1
        if self.batches_in_epoch == len(self.train_loader):
            self.data_epoch += 1
            self.batches_in_epoch = 0
            self._loader_iter = None
        return batch

    def optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema and self.scaler.get_scale() >= old_scale:
            self.ema.update(self.model)

    def save_model(self, rng_states: list, is_best: bool = False) -> None:
        """Save full precision online/EMA weights and resumable training state."""
        model = de_parallel(self.model)
        checkpoint = {
            "format_version": 2, "step": self.current_step,
            "model_config": model_config(model), "model": model.state_dict(),
            "ema": self.ema.ema.state_dict(), "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
            "best_fid_score": self.best_fid_score, "rng_states": rng_states,
            "data_epoch": self.data_epoch, "batches_in_epoch": self.batches_in_epoch,
            "resume_config": self.resume_config,
        }
        weights_path = self.save_weights_dir / f"{self.current_step}.pkl"
        temporary = weights_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(weights_path)
        for name in (("last.pkl", "best.pkl") if is_best else ("last.pkl",)):
            destination = self.save_weights_dir / name
            temporary_copy = destination.with_suffix(".tmp")
            shutil.copyfile(weights_path, temporary_copy)
            temporary_copy.replace(destination)
        self.last_saved_step = self.current_step

    def _save_checkpoint(self, is_best: bool = False) -> None:
        local_rng = self._rng_state()
        if self.world_size > 1:
            rng_states = [None] * self.world_size if self.is_main else None
            dist.gather_object(local_rng, rng_states, dst=0)
        else:
            rng_states = [local_rng]
        if self.is_main:
            self.save_model(rng_states, is_best)
        if self.world_size > 1:
            dist.barrier()

    def _sample_and_validate(self) -> bool:
        self.ema.ema.eval()
        with torch.inference_mode():
            images = torch.cat([
                self.ema.ema.sample(batch_size=n)
                for n in num_to_groups(self.num_samples, self.batch_size)
            ])
            vutils.save_image(images, self.save_samples_dir / f"{self.current_step}.png", nrow=max(1, int(math.sqrt(self.num_samples))))
        if self.fid_metrics is None:
            return False
        self.fid_score = self.fid_metrics.calculate()
        LOGGER.info(f"FID score: {self.fid_score:.2f}")
        if self.fid_score < self.best_fid_score:
            self.best_fid_score = self.fid_score
            return True
        return False

    def train(self) -> None:
        self._setup_ddp()
        try:
            self._setup_train()
            self.optimizer.zero_grad(set_to_none=True)
            with tqdm(total=self.num_steps, initial=self.current_step, disable=not self.is_main) as pbar:
                while self.current_step < self.num_steps:
                    self.model.train()
                    total_loss = 0.0
                    for microbatch in range(self.gradient_accumulate_every):
                        batch = self._next_batch().to(self.device, non_blocking=self.device.type == "cuda")
                        no_sync = self.world_size > 1 and microbatch < self.gradient_accumulate_every - 1
                        with self.model.no_sync() if no_sync else nullcontext():
                            with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                                loss = self.model(batch) / self.gradient_accumulate_every
                            total_loss += loss.detach().item()
                            self.scaler.scale(loss).backward()
                    self.optimizer_step()
                    self.current_step += 1
                    if self.is_main:
                        pbar.set_description(f"Loss: {total_loss:.4e}")
                        pbar.update(1)
                    if self.current_step % self.save_and_sample_every == 0:
                        is_best = self._sample_and_validate() if self.is_main else False
                        self._save_checkpoint(is_best)
                # Always produce a final resumable checkpoint, even between intervals.
                if self.current_step > 0 and (self.current_step % self.save_and_sample_every):
                    self._save_checkpoint()
        finally:
            if self.world_size > 1 and dist.is_initialized():
                dist.destroy_process_group()
