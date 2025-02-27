# Copyright (c) Lornatang. All rights reserved.
from ddpm.engine.trainer import Trainer
from ddpm.models.gaussian_diffusion import GaussianDiffusion
from ddpm.models.unet import UNet

model = UNet(128, channels_multipliers=(1, 2, 2, 2), dropout=0.1)
diffusion_model = GaussianDiffusion(model, 32, 1000, sampling_time_steps=250)

trainer = Trainer(
    diffusion_model,
    "/mnt/data/larry/datasets/open_data/cifar10/train",
    True,
    batch_size=128,
    gradient_accumulate_every=1,
    lr=2e-4,
    betas=(0.9, 0.999),
    num_steps=800000,
    max_grad_norm=1.0,
    val=False,
    num_samples=16,
    num_fid_samples=50000,
    fid_stat_path="./datasets/stats/cifar10_train.npz",
    save_and_sample_every=10000,
    save_dir="./results/train/cifar10",
    device="0",
)

trainer.train()
