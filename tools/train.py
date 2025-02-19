# Copyright (c) Lornatang. All rights reserved.
from ddpm.models import UNet, GaussianDiffusion
from ddpm.engine.trainer import Trainer

model = UNet(64, channels_multipliers=(1, 2, 4, 8), dropout=0.1)
diffusion_model = GaussianDiffusion(model, 32)

trainer = Trainer(
    diffusion_model,
    "/mnt/data/larry/datasets/open_data/cifar10/train",
    128,
    gradient_accumulate_every=1,
    lr=2e-4,
    num_steps=800000,
    save_dir="./results/train/cifar10",
    device="0",
)

trainer.train()
