# Copyright (c) Lornatang. All rights reserved.
from ddpm.engine.validator import Validator

validator = Validator(
    "./results/train/cifar10/weights/best_fid.pkl",
    "/mnt/data/larry/datasets/open_data/cifar10/train",
    128,
    num_samples=16,
    num_fid_samples=50000,
    fid_stat_path="./datasets/stats/cifar10_val.npz",
    save_dir="./results/val/cifar10",
    device="0",
)

validator.validate()
