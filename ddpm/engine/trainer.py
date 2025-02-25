# Copyright (c) Lornatang. All rights reserved.
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Union, Tuple

import math
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm.auto import tqdm

from ddpm.data.base import BaseImageFolder
from ddpm.data.utils import cycle_data_loader
from ddpm.evaluation.fid import FIDMetrics
from ddpm.utils.checkpoint import convert_optimizer_state_dict_to_half
from ddpm.utils.checkpoint import load_checkpoint
from ddpm.utils.ema import ModelEMA
from ddpm.utils.events import LOGGER
from ddpm.utils.ops import num_to_groups
from ddpm.utils.torch_utils import select_device


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            root: Union[Path, str],
            augment: bool = False,
            resume: bool = False,
            batch_size: int = 128,
            gradient_accumulate_every: int = 1,
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.99),
            num_steps: int = 100000,
            num_samples: int = 16,
            max_grad_norm: float = 1.0,
            num_fid_samples: int = 50000,
            fid_stat_path: Union[Path, str] = "./datasets/stats/train.npz",
            save_and_sample_every: int = 1000,
            save_dir: Union[Path, str] = "./results/train",
            device: str = "0",
    ) -> None:
        """Initialize the Trainer class.

        Args:
            model (torch.nn.Module): The diffusion model to be trained.
            root (Union[Path, str]): The root directory for the dataset.
            augment (bool, optional): Whether to apply data augmentation. Defaults to False.
            resume (bool, optional): Whether to resume training. Defaults to False.
            batch_size (int, optional): The batch size for training. Defaults to 128.
            gradient_accumulate_every (int, optional): Number of gradient accumulation steps. Defaults to 1.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            betas (Tuple[float, float], optional): Betas for the Adam optimizer. Defaults to (0.9, 0.99).
            num_steps (int, optional): Total number of training steps. Defaults to 100000.
            num_samples (int, optional): Number of samples to generate. Defaults to 16.
            fid_stat_path (Union[Path, str], optional): Path to the dataset statistics. Defaults to ``./datasets/stats/train.npz``.
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 1.0.
            num_fid_samples (int, optional): Number of samples for FID evaluation. Defaults to 50000.
            save_and_sample_every (int, optional): Frequency of saving and sampling. Defaults to 1000.
            save_dir (Union[Path, str], optional): Directory to save results. Defaults to "./results/train".
        """
        super().__init__()
        self.model = model
        self.root = root
        self.augment = augment
        self.resume = resume
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.lr = lr
        self.betas = betas
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.max_grad_norm = max_grad_norm
        self.num_fid_samples = num_fid_samples
        self.fid_stat_path = fid_stat_path
        self.save_and_sample_every = save_and_sample_every
        self.save_dir = Path(save_dir)
        self.save_samples_dir = self.save_dir.joinpath("samples")
        self.save_weights_dir = self.save_dir.joinpath("weights")
        self.save_samples_dir.mkdir(parents=True, exist_ok=True)
        self.save_weights_dir.mkdir(parents=True, exist_ok=True)
        self.device = select_device(device, self.batch_size)

        # Model
        self.model = model.to(self.device)

        # EMA
        self.ema = ModelEMA(self.model)

        # AMP
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)

        # Dataset and dataloader
        transform = transforms.Compose([
            transforms.Resize(self.model.image_size),
            transforms.CenterCrop(self.model.image_size),
            transforms.RandomHorizontalFlip() if self.augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])
        self.dataset = BaseImageFolder(self.root, transform)
        data_loader = DataLoader(self.dataset, batch_size, True, num_workers=8, pin_memory=True, persistent_workers=True)
        self.data_loader = cycle_data_loader(data_loader)

        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr, betas)

        # Resume
        self.resume_training()

        # Step counter state
        self.current_step = 0

        # Metric computation
        self.inception_score = None
        self.fid_score = None
        self.best_inception_score = 0
        self.best_fid_score = 1e10
        self.fid_metrics = FIDMetrics(
            self.data_loader,
            self.batch_size,
            self.ema.ema,
            dataset_stat_path=self.fid_stat_path,
            num_samples=self.num_fid_samples,
            device=self.device,
        )

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

        _, checkpoint = load_checkpoint(self.save_weights_dir.joinpath("last.pkl"), self.device)
        best_inception_score = 0
        best_fid_score = 10e0
        current_step = checkpoint.get("step", -1) + 1

        if checkpoint.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])  # optimizer
            best_inception_score = checkpoint["best_inception_score"]
            best_fid_score = checkpoint["best_fid_score"]

        if self.ema and checkpoint.get("ema"):
            self.ema.ema.load_state_dict(checkpoint["ema"].float().state_dict())  # EMA
            self.ema.updates = checkpoint["updates"]

        assert current_step > 0, f"Start a new training without resuming"
        LOGGER.info(f"Resuming training from epoch {current_step + 1} to {self.num_steps} total steps")
        if self.num_steps < current_step:
            LOGGER.info(f"This model has been trained for {current_step} steps. Fine-tuning for {self.num_steps} more steps.")
            self.num_steps += current_step  # finetune additional steps

        self.best_inception_score = best_inception_score
        self.best_fid_score = best_fid_score
        self.current_step = current_step

    def train(self):
        with tqdm(total=self.num_steps, initial=self.current_step) as pbar:
            while self.current_step < self.num_steps:
                self.model.train()

                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.data_loader).to(self.device)

                    # Forward
                    with torch.amp.autocast("cuda", enabled=True):
                        loss = self.model(data)
                        loss /= self.gradient_accumulate_every
                        total_loss += loss.item()

                    # Backward
                    self.scaler.scale(loss).backward()

                pbar.set_description(f"Loss: {total_loss:.4e}")

                # Update optimizer
                self.optimizer_step()

                # Save and sample
                if self.current_step != 0 and (self.current_step % self.save_and_sample_every) == 0:
                    self.ema.ema.eval()

                    with torch.no_grad():
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        all_images_list = list(map(lambda num_images: self.ema.ema.sample(batch_size=num_images), batches))

                    all_images = torch.cat(all_images_list, dim=0)
                    vutils.save_image(all_images, self.save_samples_dir.joinpath(f"{self.current_step}.png"), nrow=int(math.sqrt(self.num_samples)))

                    # Calculate Inception Score and FID
                    self.inception_score, self.fid_score = self.fid_metrics.calculate()
                    LOGGER.info(f"Inception score: {self.inception_score:.2f}, FID score: {self.fid_score:.2f}")

                    # Save model
                    self.save_model()

                pbar.update(1)

    def save_model(self) -> None:
        """Save model and ema model states to disk."""
        model_weights_path = self.save_weights_dir.joinpath(f"{self.current_step}.pkl")
        torch.save(
            {
                "step": self.current_step,
                "best_inception_score": self.best_inception_score,
                "best_fid_score": self.best_fid_score,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_half(deepcopy(self.optimizer.state_dict())),
            },
            model_weights_path,
        )
        shutil.copyfile(model_weights_path, self.save_weights_dir.joinpath("last.pkl"))
        if self.best_fid_score > self.fid_score:
            self.best_fid_score = self.fid_score
            shutil.copyfile(model_weights_path, self.save_weights_dir.joinpath("best_fid.pkl"))

        if self.best_inception_score < self.inception_score:
            self.best_inception_score = self.inception_score
            shutil.copyfile(model_weights_path, self.save_weights_dir.joinpath("best_is.pkl"))
