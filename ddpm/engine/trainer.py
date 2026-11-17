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
import itertools
import os
import random
import shutil
import uuid
import warnings
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch import distributed as dist, nn, optim
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
from torchvision import utils as vutils
from tqdm.auto import tqdm

from ddpm.data.base import BaseImageFolder, ReproducibleImageFolder, image_transform
from ddpm.evaluation.fid_metrics import FIDMetrics
from ddpm.utils.checkpoint import load_checkpoint, model_config
from ddpm.utils.ema import ModelEMA
from ddpm.utils.envs import set_seed_everything
from ddpm.utils.events import LOGGER
from ddpm.utils.ops import num_to_groups, save_yaml
from ddpm.utils.torch_utils import de_parallel


class EpochDistributedSampler(DistributedSampler):
    """Include the epoch in each index, including in persistent workers."""

    def __iter__(self):
        """Yield image indices paired with the current epoch.

        Returns:
            Iterator[Tuple[int, int]]: ``(image_index, epoch)`` pairs for this rank.
        """
        return ((index, self.epoch) for index in super().__iter__())


class ResumableBatchSampler(BatchSampler):
    """Seek in sampler indices without loading and transforming earlier images."""

    def __init__(self, sampler, batch_size, drop_last):
        """Initialize the batch sampler and its starting batch.

        Args:
            sampler (Sampler): Sampler providing individual image indices.
            batch_size (int): Number of indices per batch on this rank.
            drop_last (bool): Whether to omit the final incomplete batch.
        """
        super().__init__(sampler, batch_size, drop_last)
        self.start_batch = 0

    def __iter__(self):
        """Skip completed batches and yield the remaining index batches.

        Returns:
            Iterator[list]: Batches beginning at ``start_batch``.
        """
        return itertools.islice(super().__iter__(), self.start_batch, None)

    def __len__(self):
        """Return the number of batches remaining after ``start_batch``.

        Returns:
            int: Number of batches left to iterate.
        """
        return max(0, super().__len__() - self.start_batch)


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
        resume_checkpoint: Union[dict, None] = None,
        fid_every: Union[int, None] = None,
    ) -> None:
        """Configure distributed training, checkpointing, sampling, and FID.

        Args:
            model (nn.Module): Diffusion model with ``image_size``, a loss-producing forward method, and ``sample``.
            root (Path | str): Directory containing training images.
            augment (bool, optional): Enable horizontal flips. Defaults to ``False``.
            resume (Path | str | None, optional): Trusted checkpoint to resume. Defaults to ``None``.
            batch_size (int, optional): Global microbatch across ranks; must be divisible by world size. Defaults to 128.
            num_workers (int, optional): DataLoader workers per rank. Defaults to 8.
            gradient_accumulate_every (int, optional): Microbatches per optimizer step. Defaults to 1.
            lr (float, optional): Adam learning rate. Defaults to 1e-4.
            betas (Tuple[float, float], optional): Adam momentum coefficients. Defaults to ``(0.9, 0.999)``.
            num_steps (int, optional): Total optimizer steps, including resumed steps. Defaults to 100000.
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 1.0.
            val (bool, optional): Enable training-time FID. Defaults to ``False``.
            num_samples (int, optional): Number of images in each saved preview. Defaults to 64.
            num_fid_samples (int, optional): Number of real and generated images used for FID. Defaults to 50000.
            fid_stat_path (Path | str, optional): Real-image FID statistics cache. Defaults to ``"./datasets/stats/train.npz"``.
            save_and_sample_every (int, optional): Optimizer-step interval for checkpoints and previews. Defaults to 1000.
            save_dir (Path | str, optional): Output directory for a new run. Defaults to ``"./results/train"``.
                A resumed run uses the directory containing the checkpoint's ``weights`` directory.
            device (str, optional): CUDA device or ``"cpu"`` for testing. Defaults to ``"cuda"``; torchrun uses
                ``LOCAL_RANK`` to select each CUDA device.
            seed (int, optional): Base seed for model and data RNGs. Defaults to 0.
            fid_feature_dims (int, optional): Inception feature dimension; 2048 is standard FID. Defaults to 2048.
            resume_checkpoint (dict | None, optional): Already loaded checkpoint whose online weights are in ``model``.
                Defaults to ``None``; when ``resume`` is supplied, its checkpoint is loaded during setup.
            fid_every (int | None, optional): FID interval in optimizer steps. Defaults to ``None``, which uses
                ``save_and_sample_every``.

        Raises:
            ValueError: If distributed ranks, batch size, or training intervals are invalid.
        """
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
        if fid_every is not None and fid_every < 1:
            raise ValueError("fid_every must be positive")

        self.root = str(root)
        self.augment = augment
        self.resume = str(resume) if resume else None
        self._resume_checkpoint = resume_checkpoint
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gradient_accumulate_every = gradient_accumulate_every
        self.lr, self.betas = lr, betas
        self.num_steps, self.max_grad_norm = num_steps, max_grad_norm
        self.val, self.num_samples = val, num_samples
        self.num_fid_samples, self.fid_stat_path = num_fid_samples, str(fid_stat_path)
        self.fid_feature_dims = fid_feature_dims
        self.save_and_sample_every, self.seed = save_and_sample_every, seed
        self.fid_every = fid_every if fid_every is not None else save_and_sample_every
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
        """Select the CPU or CUDA device for this rank.

        Args:
            device (str): Device specification; distributed CUDA runs use ``LOCAL_RANK``.

        Returns:
            torch.device: Selected local device.

        Raises:
            RuntimeError: If a CUDA device is requested but CUDA is unavailable.
            ValueError: If the requested CUDA index is unavailable.
        """
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
        """Initialize NCCL on CUDA or Gloo on CPU for multiple ranks."""
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if self.device.type == "cuda" else "gloo",
                init_method="env://", timeout=timedelta(hours=12),
            )

    def _setup_train(self) -> None:
        """Build the model, data loader, optimizer, scaler, EMA, and optional FID metric.

        A resumed checkpoint selects the legacy or stateless data pipeline and
        restores training state after the corresponding objects are created.

        Raises:
            ValueError: If the checkpoint data mode is unsupported or the dataset cannot supply a complete batch.
        """
        checkpoint, self._resume_checkpoint = self._resume_checkpoint, None
        if self.resume and checkpoint is None:
            self.model, checkpoint = load_checkpoint(self.resume, self.device, model=self.model, prefer_ema=False)
        self.model = self.model.to(self.device)
        if self.world_size > 1:
            device_ids = [self.local_rank] if self.device.type == "cuda" else None
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=device_ids)

        self.data_mode = checkpoint.get("data_mode", "legacy") if checkpoint is not None else "stateless"
        if self.data_mode not in ("legacy", "stateless"):
            raise ValueError(f"Unsupported checkpoint data mode: {self.data_mode}")
        if self.data_mode == "legacy":
            dataset = BaseImageFolder(self.root, image_transform(self.image_size, augment=self.augment, normalize=True))
        else:
            dataset = ReproducibleImageFolder(
                self.root, image_transform(self.image_size, normalize=True), seed=self.seed, augment=self.augment
            )
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
        sampler_type = EpochDistributedSampler if self.data_mode == "stateless" else DistributedSampler
        self.train_sampler = sampler_type(
            dataset, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, seed=self.seed, drop_last=True,
        )
        self.data_generator = torch.Generator()
        self.batch_sampler = ResumableBatchSampler(self.train_sampler, self.batch_size // self.world_size, drop_last=True)
        self.batches_per_epoch = len(self.batch_sampler)
        self.train_loader = DataLoader(
            dataset, batch_sampler=self.batch_sampler, num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.data_mode == "stateless" and self.num_workers > 0,
            generator=self.data_generator,
        )
        if not self.batches_per_epoch:
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
        """Capture the current rank's Python, NumPy, PyTorch, and CUDA RNG states.

        Returns:
            dict: RNG states; ``cuda`` is ``None`` for a CPU run.
        """
        return {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }

    def _set_rng_state(self, state: dict) -> None:
        """Restore the current rank's saved RNG states.

        Args:
            state (dict): State produced by ``_rng_state``.
        """
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if self.device.type == "cuda" and state.get("cuda") is not None:
            torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)

    def resume_training(self, checkpoint: dict) -> None:
        """Restore optimizer, EMA, data position, and per-rank RNG state.

        Args:
            checkpoint (dict): Loaded checkpoint containing training state.

        Raises:
            ValueError: If the saved configuration, data position, or world size is incompatible.
        """
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
        if self.batches_in_epoch >= self.batches_per_epoch:
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
        """Return the next rank-local batch, restoring the saved epoch position if needed.

        Returns:
            torch.Tensor: Preprocessed image batch on the DataLoader's device.

        Note:
            Legacy checkpoints replay earlier batches to recover augmentation RNG state;
            stateless checkpoints skip sampler indices without decoding those images.
        """
        if self._loader_iter is None:
            self.train_sampler.set_epoch(self.data_epoch)
            self.data_generator.manual_seed(self.seed + self.data_epoch * self.world_size + self.rank)
            self.batch_sampler.start_batch = self.batches_in_epoch if self.data_mode == "stateless" else 0
            self._loader_iter = iter(self.train_loader)
            # Rebuild the same worker/sampler position without consuming model RNG.
            if self.batches_in_epoch and self.data_mode == "legacy":
                state = self._rng_state()
                for _ in range(self.batches_in_epoch):
                    next(self._loader_iter)
                self._set_rng_state(state)
        batch = next(self._loader_iter)
        self.batches_in_epoch += 1
        if self.batches_in_epoch == self.batches_per_epoch:
            self.data_epoch += 1
            self.batches_in_epoch = 0
            self._loader_iter = None
        return batch

    def optimizer_step(self) -> None:
        """Apply a clipped AMP optimizer step and update EMA when the step succeeds."""
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema and self.scaler.get_scale() >= old_scale:
            self.ema.update(self.model)

    def save_model(self, rng_states: list, is_best: bool = False) -> None:
        """Save full-precision model weights and resumable training state.

        Args:
            rng_states (list[dict]): RNG state from each rank in rank order.
            is_best (bool, optional): Update the best-FID checkpoint alias. Defaults to ``False``.

        Note:
            Checkpoint aliases use hard links when supported and fall back to copies.
        """
        model = de_parallel(self.model)
        checkpoint = {
            "format_version": 3, "step": self.current_step,
            "model_config": model_config(model), "model": model.state_dict(),
            "ema": self.ema.ema.state_dict(), "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
            "best_fid_score": self.best_fid_score, "rng_states": rng_states,
            "data_epoch": self.data_epoch, "batches_in_epoch": self.batches_in_epoch, "data_mode": self.data_mode,
            "resume_config": self.resume_config,
        }
        weights_path = self.save_weights_dir / f"{self.current_step}.pkl"
        temporary = weights_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(weights_path)
        for name in (("last.pkl", "best.pkl") if is_best else ("last.pkl",)):
            destination = self.save_weights_dir / name
            temporary_copy = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                try:
                    os.link(weights_path, temporary_copy)
                except OSError:
                    shutil.copyfile(weights_path, temporary_copy)
                temporary_copy.replace(destination)
            finally:
                temporary_copy.unlink(missing_ok=True)
        self.last_saved_step = self.current_step

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Gather rank-local RNG states and save a checkpoint on rank 0.

        Args:
            is_best (bool, optional): Update the best-FID checkpoint alias. Defaults to ``False``.
        """
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

    def _save_samples(self) -> None:
        """Save an EMA-generated image grid for the current step on rank 0."""
        self.ema.ema.eval()
        with torch.inference_mode():
            images = torch.cat([
                self.ema.ema.sample(batch_size=n)
                for n in num_to_groups(self.num_samples, self.batch_size)
            ])
            vutils.save_image(images, self.save_samples_dir / f"{self.current_step}.png", nrow=max(1, int(math.sqrt(self.num_samples))))

    @torch.no_grad()
    def _evaluate(self) -> bool:
        """Calculate FID, synchronizing EMA weights and generated features across ranks.

        Returns:
            bool: ``True`` on rank 0 if the score improves the best FID; ``False`` otherwise.
        """
        if self.world_size == 1:
            score = self.fid_metrics.calculate()
        else:
            # Only rank 0 maintains EMA during training. Broadcast its current
            # state in dtype groups before every distributed evaluation.
            model = self.ema.ema if self.is_main else deepcopy(de_parallel(self.model)).eval().requires_grad_(False)
            groups = {}
            for value in model.state_dict().values():
                groups.setdefault((value.device, value.dtype), []).append(value)
            for values in groups.values():
                flattened = torch.cat([value.reshape(-1) for value in values])
                dist.broadcast(flattened, src=0)
                if not self.is_main:
                    offset = 0
                    for value in values:
                        value.copy_(flattened[offset:offset + value.numel()].view_as(value))
                        offset += value.numel()
            if self.fid_metrics is None:
                # Rank 0 constructed Inception at setup, so its first download
                # has finished before other ranks reach this broadcast.
                self.fid_metrics = FIDMetrics(
                    None, self.batch_size, model, dataset_stat_path=self.fid_stat_path,
                    num_samples=self.num_fid_samples, device=self.device, feature_dims=self.fid_feature_dims,
                )
            self.fid_metrics.model = model
            score = self.fid_metrics.calculate_distributed()
            if not self.is_main:
                self.fid_metrics.model = None
        self.fid_score = score
        if self.is_main:
            LOGGER.info(f"FID score: {score:.2f}")
            if score < self.best_fid_score:
                self.best_fid_score = score
                return True
        return False

    def train(self) -> None:
        """Run optimizer steps, scheduled sampling, FID, and checkpointing.

        Distributed workers participate in evaluation and checkpoint collectives;
        the process group is destroyed when training finishes or fails.
        """
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
                    sample_due = self.current_step % self.save_and_sample_every == 0
                    fid_due = self.val and self.current_step % self.fid_every == 0
                    if sample_due and self.is_main:
                        self._save_samples()
                    is_best = self._evaluate() if fid_due else False
                    if sample_due or fid_due or self.current_step == self.num_steps:
                        self._save_checkpoint(is_best)
        finally:
            if self.world_size > 1 and dist.is_initialized():
                dist.destroy_process_group()
