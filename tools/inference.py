# Copyright (c) Lornatang. All rights reserved.
from ddpm.engine.inference import Inferencer

inferencer = Inferencer(
    "./results/train/cifar10/weights/best_fid.pkl",
    save_dir="./results/inference/cifar10",
    device="0",
)
inferencer.inference()
