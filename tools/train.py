# Copyright (c) Lornatang. All rights reserved.
from ddpm.models import UNet, GaussianDiffusion
from ddpm.engine.trainer import Trainer

model = UNet(64, channels_multipliers=(1, 2, 4, 8))
diffusion_model = GaussianDiffusion(model, 32)

trainer = Trainer(
    diffusion_model,
    "/mnt/data/larry/datasets/open_data/cifar10/train",
    32,
    gradient_accumulate_every=2,
    lr=8e-5,
    num_steps=700000,
)

trainer.train()