# Copyright (c) Lornatang. All rights reserved.
from datetime import datetime

from ddpm.engine.validator import Validator

validator = Validator(
    "./results/train/cifar10/weights/best.pkl",
    "/mnt/data/larry/datasets/open_data/cifar10/train",
    256,
    num_fid_samples=50000,
    fid_stat_path="./datasets/stats/cifar10_train.npz",
    save_dir=f"./results/val/cifar10-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    device="0",
)
validator.validate()
