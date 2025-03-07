# Copyright (c) Lornatang. All rights reserved.
from datetime import datetime

from ddpm.engine.trainer import Trainer
from ddpm.models.gaussian_diffusion import GaussianDiffusion
from ddpm.models.unet import UNet

model = UNet(128, channels_multipliers=(1, 2, 2, 2), flash=True)
diffusion_model = GaussianDiffusion(model, 256, 1000, sampling_time_steps=250)

trainer = Trainer(
    diffusion_model,
    "/mnt/data/larry/datasets/open_data/celeba/img_align_celeba",
    False,
    resume=False,
    batch_size=64,
    gradient_accumulate_every=1,
    lr=2e-5,
    betas=(0.9, 0.999),
    num_steps=500000,
    max_grad_norm=1.0,
    val=False,
    num_samples=16,
    num_fid_samples=50000,
    fid_stat_path="./datasets/stats/celeba_train_r256.npz",
    save_and_sample_every=10000,
    save_dir=f"./results/train/celeba-r256-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    device="0",
    seed=0,
)

trainer.train()
