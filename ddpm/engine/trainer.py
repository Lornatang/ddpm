# Copyright (c) Lornatang. All rights reserved.
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
from ddpm.utils.ema import ModelEMA
from ddpm.utils.events import LOGGER
from ddpm.utils.ops import num_to_groups
from ddpm.utils.torch_utils import select_device


class Trainer:
    def __init__(
            self,
            diffusion_model: torch.nn.Module,
            root: Union[Path, str],
            batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.99),
            num_steps: int = 100000,
            num_samples: int = 16,
            max_grad_norm: float = 1.0,
            num_fid_samples: int = 50000,
            fid_stat_path: Union[Path, str] = "./datasets/stats/train.npz",
            save_and_sample_every: int = 1000,
            save_dir: Union[Path, str] = "./results",
            device: str = "0",
    ) -> None:
        """Initialize the Trainer class.

        Args:
            diffusion_model (torch.nn.Module): The diffusion model to be trained.
            root (Union[Path, str]): The root directory for the dataset.
            batch_size (int, optional): The batch size for training. Defaults to 16.
            gradient_accumulate_every (int, optional): Number of gradient accumulation steps. Defaults to 1.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            betas (Tuple[float, float], optional): Betas for the Adam optimizer. Defaults to (0.9, 0.99).
            num_steps (int, optional): Total number of training steps. Defaults to 100000.
            num_samples (int, optional): Number of samples to generate. Defaults to 16.
            fid_stat_path (Union[Path, str], optional): Path to the dataset statistics. Defaults to ``./datasets/stats/train.npz``.
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 1.0.
            num_fid_samples (int, optional): Number of samples for FID evaluation. Defaults to 50000.
            save_and_sample_every (int, optional): Frequency of saving and sampling. Defaults to 1000.
            save_dir (Union[Path, str], optional): Directory to save results. Defaults to "./results".
        """
        super().__init__()
        self.diffusion_model = diffusion_model
        self.channels = diffusion_model.in_channels
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
        self.diffusion_model = diffusion_model.to(self.device)

        # AMP
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)

        # dataset and dataloader
        transform = transforms.Compose([
            transforms.Resize(self.diffusion_model.image_size),
            transforms.CenterCrop(self.diffusion_model.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.dataset = BaseImageFolder(root, transform)
        data_loader = DataLoader(self.dataset, batch_size, True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
        self.data_loader = cycle_data_loader(data_loader)

        # optimizer
        self.optimizer = Adam(self.diffusion_model.parameters(), lr, betas)

        # ema
        self.ema = ModelEMA(self.diffusion_model)

        # step counter state
        self.step = 0

        # FID-score computation
        self.best_fid = 1e10
        self.fid_scorer = FIDMetrics(
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
        nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), max_norm=1.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.diffusion_model)
        self.step += 1

    def save(self, milestone: str) -> None:
        """Save model and ema model states to disk.

        Args:
            milestone (str): milestone name
        """
        ckpt = {
            "step": self.step,
            "best_fid": self.best_fid,
            "model": None,  # resume and final checkpoints derive from EMA
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": convert_optimizer_state_dict_to_half(deepcopy(self.optimizer.state_dict())),
        }
        torch.save(ckpt, str(self.save_weights_dir.joinpath(f"{milestone}.pkl")))

    def train(self):
        with tqdm(total=self.num_steps, initial=self.step) as pbar:
            while self.step < self.num_steps:
                self.diffusion_model.train()

                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.data_loader).to(self.device)

                    # Forward
                    with torch.amp.autocast("cuda", enabled=True):
                        loss = self.diffusion_model(data)
                        loss /= self.gradient_accumulate_every
                        total_loss += loss.item()

                    # Backward
                    self.scaler.scale(loss).backward()

                pbar.set_description(f"Loss: {total_loss:.4e}")

                # Update optimizer
                self.optimizer_step()

                # Save and sample
                if self.step != 0 and (self.step % self.save_and_sample_every) == 0:
                    self.ema.ema.eval()

                    with torch.no_grad():
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        all_images_list = list(map(lambda num_images: self.ema.ema.sample(batch_size=num_images), batches))

                    all_images = torch.cat(all_images_list, dim=0)
                    vutils.save_image(all_images, self.save_samples_dir.joinpath(f"{self.step}.png"), nrow=int(math.sqrt(self.num_samples)))

                    # Cal fid and save model
                    fid_score = self.fid_scorer.fid_score()
                    LOGGER.info(f"FID score: {fid_score:.2f}")

                    if self.best_fid > fid_score:
                        self.best_fid = fid_score
                        self.save(str(self.step))
                        self.save("best")
                    self.save("last")

                pbar.update(1)
