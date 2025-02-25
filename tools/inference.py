# Copyright (c) Lornatang. All rights reserved.
from ddpm.engine.inference import Inferencer

inferencer = Inferencer(
    "./results/train/new_cifar10/weights/last.pkl",
    save_dir="./results/inference/new_cifar10",
    device="0",
)
inferencer.inference()
