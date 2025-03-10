# Copyright (c) Lornatang. All rights reserved.
import gc
import os
import shutil
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Union

import math
import torch
from torch import distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm.auto import tqdm

from ddpm.data.base import BaseImageFolder
from ddpm.data.utils import cycle_data_loader
from ddpm.evaluation.fid_metrics import FIDMetrics
from ddpm.utils.checkpoint import convert_optimizer_state_dict_to_half, load_checkpoint
from ddpm.utils.ema import ModelEMA
from ddpm.utils.envs import LOCAL_RANK, RANK, set_seed_everything, unset_deterministic
from ddpm.utils.events import LOGGER
from ddpm.utils.ops import num_to_groups, save_yaml
from ddpm.utils.torch_utils import select_device


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            root: Union[Path, str],
            augment: bool = False,
            resume: Union[Path, str] = None,
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
            device: str = "0",
            seed: int = 0,
    ) -> None:
        """Initialize the Trainer class.

        Args:
            model (torch.nn.Module): The diffusion model to be trained.
            root (Union[Path, str]): The root directory for the dataset.
            augment (bool, optional): Whether to apply data augmentation. Defaults to ``False``.
            resume (Union[Path, str], optional): Whether to resume training. Defaults to ``None``.
            batch_size (int, optional): The batch size for training. Defaults to 128.
            gradient_accumulate_every (int, optional): Number of gradient accumulation steps. Defaults to 1.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            betas (Tuple[float, float], optional): Betas for the Adam optimizer. Defaults to (0.9, 0.999).
            num_steps (int, optional): Total number of training steps. Defaults to 100000.
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 1.0.
            val (bool, optional): Whether to use validation. Defaults to ``False``.
            num_samples (int, optional): Number of samples to generate. Defaults to 64.
            fid_stat_path (Union[Path, str], optional): Path to the dataset statistics. Defaults to ``./datasets/stats/train.npz``.
            num_fid_samples (int, optional): Number of samples for FID evaluation. Defaults to 50000.
            save_and_sample_every (int, optional): Frequency of saving and sampling. Defaults to 1000.
            save_dir (Union[Path, str], optional): Directory to save results. Defaults to ``./results/train``.
            device (str, optional): Device to use. Defaults to ``0``.
            seed (int, optional): Random seed. Defaults to ``0``.
        """
        super().__init__()
        self.root = root
        self.augment = augment
        self.resume = resume
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gradient_accumulate_every = gradient_accumulate_every
        self.lr = lr
        self.betas = betas
        self.num_steps = num_steps
        self.max_grad_norm = max_grad_norm
        self.val = val
        self.num_samples = num_samples
        self.num_fid_samples = num_fid_samples
        self.fid_stat_path = fid_stat_path
        self.save_and_sample_every = save_and_sample_every
        self.device_type = device
        self.device = select_device(self.device_type, self.batch_size)
        self.seed = seed

        # Set seed
        set_seed_everything(seed + 1 + RANK)

        # Create save directories
        if not self.resume:
            self.save_dir = Path(save_dir)
            self.save_samples_dir = self.save_dir.joinpath("samples")
            self.save_weights_dir = self.save_dir.joinpath("weights")
            if RANK in {-1, 0}:
                self.save_samples_dir.mkdir(parents=True, exist_ok=True)
                self.save_weights_dir.mkdir(parents=True, exist_ok=True)
                save_yaml(Path(self.save_dir, "opts.yaml"), vars(self))
        else:
            self.save_dir = Path(self.resume).parent.parent
            self.save_samples_dir = self.save_dir.joinpath("samples")
            self.save_weights_dir = self.save_dir.joinpath("weights")

        # Step counter state
        self.current_step = 0

        # Model
        self.model = model
        self.image_size = self.model.image_size
        self.ema = None

        # Optimizer
        self.optimizer = None

        # Metric computation
        self.fid_metrics = None
        self.fid_score = None
        self.best_fid_score = 1e10

    def optimizer_step(self) -> None:
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        self.current_step += 1

    def resume_training(self) -> None:
        """Resume training from a checkpoint."""
        if not self.resume:
            return

        _, checkpoint = load_checkpoint(self.resume, self.device)
        self.current_step = checkpoint.get("step", -1) + 1

        if checkpoint.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])  # optimizer
            self.best_fid_score = checkpoint["best_fid_score"]

        if self.ema and checkpoint.get("ema"):
            self.ema.ema.load_state_dict(checkpoint["ema"].float().state_dict())  # EMA
            self.ema.updates = checkpoint["updates"]

        assert self.current_step > 0, f"Start a new training without resuming"
        LOGGER.info(f"Resuming training from epoch {self.current_step + 1} to {self.num_steps} total steps")
        if self.num_steps < self.current_step:
            LOGGER.info(f"This model has been trained for {self.current_step} steps. Fine-tuning for {self.num_steps} more steps.")
            self.num_steps += self.current_step  # finetune additional steps

    def get_validater(self) -> FIDMetrics:
        return FIDMetrics(
            self.data_loader,
            self.batch_size * 2,
            self.ema.ema,
            dataset_stat_path=self.fid_stat_path,
            num_samples=self.num_fid_samples,
            device=self.device,
        )

    def train(self) -> None:
        if isinstance(self.device_type, str) and len(self.device_type):
            world_size = len(self.device_type.split(","))
        elif isinstance(self.device_type, (tuple, list)):
            world_size = len(self.device_type)
        elif torch.cuda.is_available():
            world_size = 1
        else:
            world_size = 0

        self._do_train(world_size)

    def _clear_memory(self):
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def _setup_ddp(self, world_size: int = 1) -> None:
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=7200000),
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size: int) -> None:
        # Model
        self.model = self.model.to(self.device)

        # AMP
        self.amp = torch.tensor(True).to(self.device)
        if RANK > -1 and world_size > 1:
            dist.broadcast(self.amp, src=0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(self.amp))

        # DDP
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[LOCAL_RANK])

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.RandomHorizontalFlip() if self.augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = BaseImageFolder(self.root, transform)
        data_loader = DataLoader(dataset, batch_size, True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
        self.data_loader = cycle_data_loader(data_loader)
        if RANK in {-1, 0}:
            self.ema = ModelEMA(self.model)
            if self.val:
                self.fid_metrics = self.get_validater()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), self.lr, self.betas)

        # Resume
        self.resume_training()

    def _do_train(self, world_size: int = 1) -> None:
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        with tqdm(total=self.num_steps, initial=self.current_step) as pbar:
            while self.current_step < self.num_steps:
                self.model.train()

                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    batch_data = next(self.data_loader).to(self.device)

                    # Forward
                    with torch.amp.autocast("cuda", enabled=bool(self.amp)):
                        loss = self.model(batch_data)
                        loss /= self.gradient_accumulate_every
                        if RANK != -1:
                            loss *= world_size
                        total_loss += loss.item()

                    # Backward
                    self.scaler.scale(loss).backward()

                # Update optimizer
                self.optimizer_step()

                # Log and save
                if RANK in {-1, 0}:
                    pbar.set_description(f"Loss: {total_loss:.4e}")
                    pbar.update(1)

                    # Save and sample
                    if self.current_step != 0 and (self.current_step % self.save_and_sample_every) == 0:
                        self.ema.ema.eval()

                        with torch.no_grad():
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda num_images: self.ema.ema.sample(batch_size=num_images), batches))

                        all_images = torch.cat(all_images_list, dim=0)
                        vutils.save_image(
                            all_images,
                            Path(self.save_samples_dir, f"{self.current_step}.png"),
                            nrow=int(math.sqrt(self.num_samples))
                        )

                        if self.val:
                            # Calculate Inception Score and FID
                            self.fid_score = self.fid_metrics.calculate()
                            LOGGER.info(f"FID score: {self.fid_score:.2f}")

                        # Save model
                        self.save_model()

        # End
        if RANK in {-1, 0}:
            self._clear_memory()
            unset_deterministic()

    def save_model(self) -> None:
        """Save model and ema model states to disk."""
        model_weights_path = Path(self.save_weights_dir, f"{self.current_step}.pkl")
        torch.save(
            {
                "step": self.current_step,
                "best_fid_score": self.best_fid_score,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_half(deepcopy(self.optimizer.state_dict())),
            },
            model_weights_path,
        )
        shutil.copyfile(model_weights_path, Path(self.save_weights_dir, "last.pkl"))

        if self.val:
            if self.best_fid_score > self.fid_score:
                self.best_fid_score = self.fid_score
                shutil.copyfile(model_weights_path, Path(self.save_weights_dir, "best.pkl"))
