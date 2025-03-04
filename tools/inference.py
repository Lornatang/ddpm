# Copyright (c) Lornatang. All rights reserved.
from datetime import datetime

from ddpm.engine.inference import Inferencer

inferencer = Inferencer(
    "./results/train/cifar10/weights/last.pkl",
    save_dir=f"./results/inference/cifar10-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    device="0",
)
inferencer.inference()
