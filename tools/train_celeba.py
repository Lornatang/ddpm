# Copyright (c) Lornatang. All rights reserved.
from ddpm.engine.trainer import Trainer
from ddpm.models.gaussian_diffusion import GaussianDiffusion
from ddpm.models.unet import UNet

model = UNet(128, channels_multipliers=(1, 2, 2, 2))
diffusion_model = GaussianDiffusion(model, 256, 1000, sampling_time_steps=250)

trainer = Trainer(
    diffusion_model,
    "/mnt/data/larry/datasets/open_data/celeba/img_align_celeba",
    False,
    batch_size=64,
    lr=2e-5,
    betas=(0.9, 0.99),
    num_steps=500000,
    num_samples=16,
    max_grad_norm=1.0,
    num_fid_samples=50000,
    fid_stat_path="./datasets/stats/celeba_train.npz",
    save_and_sample_every=20000,
    save_dir="./results/train/celeba",
    device="0",
)

trainer.train()
