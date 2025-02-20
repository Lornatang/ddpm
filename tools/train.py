# Copyright (c) Lornatang. All rights reserved.
from ddpm.models import UNet, GaussianDiffusion
from ddpm.engine.trainer import Trainer

model = UNet(128, channels_multipliers=(1, 2, 2, 2), dropout=0.1)
diffusion_model = GaussianDiffusion(model, 32, 1000, sampling_time_steps=250)

trainer = Trainer(
    diffusion_model,
    "/mnt/data/larry/datasets/open_data/cifar10/train",
    128,
    lr=2e-4,
    betas=(0.9, 0.99),
    num_steps=800000,
    num_samples=16,
    max_grad_norm=1.0,
    num_fid_samples=50000,
    fid_stat_path="./datasets/stats/cifar10_train.npz",
    save_and_sample_every=10000,
    save_dir="./results/train/new_cifar10",
    device="0",
)

trainer.train()
