# Copyright Larry. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Command-line entry points for training, sampling and evaluation."""

import argparse
import sys

from ddpm.engine.inference import Inferencer
from ddpm.engine.trainer import Trainer
from ddpm.engine.validator import Validator
from ddpm.models.gaussian_diffusion import GaussianDiffusion
from ddpm.models.unet import UNet
from ddpm.utils.checkpoint import load_checkpoint
from ddpm.utils.envs import set_seed_everything


def train_main(argv=None, *, default_preset="cifar10"):
    """Parse training options and launch a single-process or torchrun trainer.

    Args:
        argv (list[str] | None, optional): Command-line arguments without the program name. Defaults to ``None``,
            which reads arguments from ``sys.argv``.
        default_preset (str, optional): Preset used when ``--preset`` is omitted. Defaults to ``"cifar10"``.
    """
    parser = argparse.ArgumentParser(description="Train DDPM with torchrun")
    parser.add_argument("--data", required=True, help="Directory containing training images")
    parser.add_argument("--preset", choices=("cifar10", "celeba"), default=default_preset)
    parser.add_argument("--save-dir", default="./results/train")
    parser.add_argument("--resume", help="Resume from a trusted checkpoint")
    parser.add_argument("--device", default="cuda", help="cuda (default), cuda:N, or cpu for testing")
    parser.add_argument("--batch-size", type=int, help="Global batch per microstep, across all workers")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, help="Total optimizer steps, including any resumed steps")
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--validate", action="store_true", help="Enable distributed training-time FID")
    parser.add_argument("--fid-every", type=int, help="FID interval in steps (defaults to --save-every)")
    parser.add_argument("--fid-samples", type=int, default=50000)
    parser.add_argument("--fid-stat-path", default="./datasets/stats/train.npz")
    parser.add_argument("--fid-feature-dims", type=int, choices=(64, 192, 768, 2048), default=2048,
                        help="Use 2048 for standard, comparable FID")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 2, 2])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim-head", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-steps", type=int, default=250)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--flash", action=argparse.BooleanOptionalAction, default=True,
                        help="Use PyTorch SDPA with automatic backend selection")
    parser.add_argument("--self-condition", action="store_true")
    parser.add_argument("--learned-variance", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    preset = {
        "cifar10": dict(image_size=32, batch_size=128, steps=800000, lr=2e-4,
                        augment=True, num_samples=64, dropout=0.1),
        "celeba": dict(image_size=256, batch_size=64, steps=500000, lr=2e-5,
                       augment=False, num_samples=16, dropout=0.0),
    }[args.preset]

    set_seed_everything(args.seed)
    checkpoint = None
    if args.resume:
        diffusion, checkpoint = load_checkpoint(args.resume, "cpu", prefer_ema=False)
    else:
        model = UNet(args.dim, channels_multipliers=tuple(args.multipliers), dropout=preset["dropout"],
                     attn_heads=args.heads, attn_dim_head=args.dim_head, flash=args.flash,
                     self_condition=args.self_condition, learned_variance=args.learned_variance)
        diffusion = GaussianDiffusion(model, args.image_size if args.image_size is not None else preset["image_size"], args.timesteps,
                                      sampling_time_steps=args.sampling_steps)
    trainer = Trainer(
        diffusion, args.data, augment=preset["augment"] if args.augment is None else args.augment,
        resume=args.resume, batch_size=args.batch_size if args.batch_size is not None else preset["batch_size"], num_workers=args.workers,
        gradient_accumulate_every=args.accumulate, lr=args.lr if args.lr is not None else preset["lr"],
        num_steps=args.steps if args.steps is not None else preset["steps"], max_grad_norm=args.max_grad_norm,
        val=args.validate, num_samples=args.num_samples if args.num_samples is not None else preset["num_samples"],
        num_fid_samples=args.fid_samples, fid_stat_path=args.fid_stat_path,
        save_and_sample_every=args.save_every, save_dir=args.save_dir, device=args.device, seed=args.seed,
        fid_feature_dims=args.fid_feature_dims,
        resume_checkpoint=checkpoint,
        fid_every=args.fid_every,
    )
    del checkpoint
    trainer.train()


def inference_main(argv=None):
    """Parse sampling options and generate images from a checkpoint, preferring EMA weights.

    Args:
        argv (list[str] | None, optional): Command-line arguments without the program name. Defaults to ``None``,
            which reads arguments from ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description="Generate images from a DDPM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="./results/inference")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    Inferencer(args.checkpoint, args.batch_size, args.num_samples, args.output, args.device).inference()


def validate_main(argv=None):
    """Parse evaluation options and calculate FID for a checkpoint, preferring EMA weights.

    Args:
        argv (list[str] | None, optional): Command-line arguments without the program name. Defaults to ``None``,
            which reads arguments from ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description="Evaluate a DDPM checkpoint using FID")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fid-samples", type=int, default=50000)
    parser.add_argument("--fid-stat-path", default="./datasets/stats/val.npz")
    parser.add_argument("--output", default="./results/val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--feature-dims", type=int, choices=(64, 192, 768, 2048), default=2048,
                        help="FID feature dimension; 2048 is the standard")
    args = parser.parse_args(argv)
    Validator(args.checkpoint, args.data, args.batch_size, args.fid_samples, args.fid_stat_path,
              args.output, args.device, num_workers=args.workers, feature_dims=args.feature_dims).validate()


def main():
    """Dispatch the ``train``, ``sample``, or ``validate`` subcommand.

    Raises:
        SystemExit: If the subcommand is missing or unsupported.
    """
    commands = {"train": train_main, "sample": inference_main, "validate": validate_main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit("Usage: python -m ddpm.cli {train,sample,validate} [options]")
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
