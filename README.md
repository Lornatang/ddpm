# DDPM

[![Tests](https://github.com/Lornatang/ddpm/actions/workflows/tests.yml/badge.svg)](https://github.com/Lornatang/ddpm/actions/workflows/tests.yml)

A PyTorch implementation of unconditional image generation with a time-conditioned U-Net and diffusion models. It supports distributed training on Ubuntu, DDPM and DDIM sampling, and Fréchet Inception Distance (FID) evaluation.

## Features

- Single-node, multi-GPU training with PyTorch DistributedDataParallel, `torchrun`, and CUDA automatic mixed precision.
- A time-conditioned U-Net trained with a velocity-prediction objective.
- DDPM and DDIM sampling, EMA weights, optional self-conditioning, and learned variance.
- Resumable checkpoints with optimizer, AMP scaler, per-rank RNG, and data-loader position.
- Optional distributed FID evaluation with cached real-image statistics.
- PyTorch scaled dot-product attention (SDPA) with automatic backend selection.

## Requirements

- Ubuntu and NVIDIA GPUs for training. CPU execution is available for smoke tests.
- Python 3.10 or later.
- PyTorch 2.6 or later and torchvision 0.21 or later. Install CUDA-enabled builds that match your environment.
- Enough image files for at least one global training batch; FID additionally requires at least as many real images as `--fid-samples`.

## Installation

```bash
git clone https://github.com/Lornatang/ddpm.git
cd ddpm

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch and the matching torchvision using the command from the [PyTorch installation guide](https://pytorch.org/get-started/locally/). Then install this project:

```bash
python -m pip install -e .

# Optional: install the test dependencies.
python -m pip install -e '.[test]'
```

Check that PyTorch can see the GPUs you intend to use:

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
```

## Data

Provide a directory of image files using `--data`. Images are discovered recursively; class directories are allowed, but their labels are ignored. For example:

```text
datasets/cifar10/train/
├── airplane/
│   ├── 00001.png
│   └── 00002.png
└── automobile/
    └── 00003.png
```

Images are converted to RGB, resized, and center-cropped. Training inputs are normalized to `[-1, 1]`; FID uses unaugmented real and generated images in `[0, 1]`. The `cifar10` and `celeba` names select training presets: they do not download or unpack those datasets. Prepare image files before running the commands below.

The original [CIFAR-10 distribution](https://www.cs.toronto.edu/~kriz/cifar.html) uses batch files, which need to be exported as images for this loader. The [CelebA dataset page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) provides the corresponding dataset information and download instructions.

For example, after installation, the following downloads CIFAR-10 and exports its 50,000 training images as PNG files. It expects `datasets/cifar10/train` not to exist yet:

```bash
python - <<'PY'
from pathlib import Path
from torchvision.datasets import CIFAR10

output = Path("datasets/cifar10/train")
if output.exists():
    raise FileExistsError(f"{output} already exists")
dataset = CIFAR10(root="datasets/cifar10/raw", train=True, download=True)
output.mkdir(parents=True, exist_ok=False)
for index, (image, _) in enumerate(dataset):
    image.save(output / f"{index:05d}.png")
print(f"Exported {len(dataset)} images to {output}")
PY
```

## Training

Launch one process per visible GPU on a single Ubuntu machine. For example, the following starts a two-GPU CIFAR-10 preset run for 10,000 optimizer steps:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli train \
  --data ./datasets/cifar10/train \
  --preset cifar10 \
  --batch-size 128 \
  --steps 10000 \
  --save-every 10000 \
  --save-dir ./results/cifar10
```

`--batch-size` is the **global microbatch** across all processes and must be divisible by the number of processes. The effective optimizer batch is `--batch-size * --accumulate`. For example, with two GPUs, `--batch-size 128 --accumulate 2` processes 64 images per GPU per microstep and 256 images per optimizer step. Incomplete batches are dropped.

### Two-GPU smoke test

Before a long run, use at least four images and a small model to check the training and checkpoint paths:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli train --data ./datasets/cifar10/train \
  --batch-size 4 --workers 0 --steps 2 --save-every 2 --num-samples 2 \
  --image-size 8 --dim 8 --multipliers 1 2 --heads 1 --dim-head 4 \
  --timesteps 32 --sampling-steps 3 --save-dir ./results/smoke
```

This run checks the workflow and does not produce a trained model of useful quality.

### Presets and options

| Preset | Image size | Global batch | Total steps | Learning rate | Horizontal flip |
| --- | ---: | ---: | ---: | ---: | --- |
| `cifar10` | 32 | 128 | 800,000 | 0.0002 | Enabled |
| `celeba` | 256 | 64 | 500,000 | 0.00002 | Disabled |

The table shows CLI defaults when the corresponding options are omitted. Select the CelebA preset with `--preset celeba`; paths still need to point to your own images.

Common options include `--steps`, `--lr`, `--workers`, `--accumulate`, `--image-size`, `--dim`, `--timesteps`, `--sampling-steps`, and `--save-every`. `--workers` is the number of data-loader workers **per rank**. The linear schedule requires `--timesteps` greater than 20 and `--sampling-steps` between 1 and `--timesteps`. A smaller sampling step count selects DDIM; equal counts select DDPM. Use `--augment` or `--no-augment` to override the preset, and `--self-condition` or `--learned-variance` to enable those model options. `--flash` enables SDPA with automatic backend selection, while `--no-flash` uses explicit attention. See all options with:

```bash
python -m ddpm.cli train --help
```

### Outputs

The example run produces files under `./results/cifar10`:

```text
results/cifar10/
├── opts.yaml
├── samples/
│   └── 10000.png       # EMA-generated preview grid
└── weights/
    ├── 10000.pkl       # Numbered checkpoint
    ├── last.pkl        # Latest checkpoint
    └── best.pkl        # Best FID checkpoint, when FID is enabled
```

The final optimizer step is checkpointed even if it falls between preview intervals. `last.pkl` and `best.pkl` are hard links to numbered checkpoints when supported, with a copy fallback. Do not edit checkpoint files in place because hard-linked names share the same file contents.

### Training-time FID

FID is disabled by default. To enable it, add `--validate` and configure the sample count and evaluation interval, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli train --data ./datasets/cifar10/train --batch-size 128 \
  --save-dir ./results/cifar10 --validate \
  --fid-samples 50000 --fid-every 50000 \
  --fid-stat-path ./datasets/stats/cifar10-train.npz
```

All ranks generate samples from the rank-0 EMA model. Training pauses during evaluation; rank 0 prepares real-image statistics and calculates the final score. If the pretrained Inception weights are not already cached, the first evaluation downloads them. Real-image statistics are also cached. `--fid-every` is measured in optimizer steps and defaults to `--save-every`; an evaluation also saves a checkpoint.

## Resume training

Pass the checkpoint path and the new **total** step count. For the 10,000-step example above, the following continues to step 20,000:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli train --data ./datasets/cifar10/train --batch-size 128 \
  --resume ./results/cifar10/weights/last.pkl --steps 20000
```

Use the same world size, device type, dataset path and filenames, global batch size, worker count, augmentation, gradient accumulation, gradient clipping, and seed as the saved run. The model configuration and optimizer state are restored from the checkpoint. Outputs continue in the checkpoint's run directory.

Current checkpoints restore per-rank RNG state and sampler position. Older checkpoints remain loadable, but missing RNG state or previously quantized optimizer state cannot be recovered. Load checkpoints only from trusted sources.

## Sampling

Sampling prefers EMA weights and writes individual PNG images. This example divides 100 generated images across two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli sample \
  --checkpoint ./results/cifar10/weights/last.pkl \
  --num-samples 100 --batch-size 64 --output ./results/generated
```

For sampling and FID, each rank uses `max(1, batch_size // number_of_ranks)` images per batch. Choose a batch size divisible by the rank count to use the requested global batch. For single-GPU use, run `ddpm-sample --help`.

## Evaluation

Evaluate a checkpoint against a directory of real images:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli validate \
  --checkpoint ./results/cifar10/weights/last.pkl \
  --data ./datasets/cifar10/train \
  --batch-size 128 --fid-samples 50000 --feature-dims 2048 \
  --fid-stat-path ./datasets/stats/cifar10-eval.npz \
  --output ./results/evaluation
```

FID defaults to 2,048 Inception features. Other `--feature-dims` values are useful for smoke tests but are **not comparable** to standard FID. Comparisons also require matching the real dataset, preprocessing, sample count, and sampling settings. At least `--fid-samples` real image files are required; the loader uses images in sorted path order.

The cache signature includes dataset and preprocessing metadata, image paths, file sizes, modification times, feature dimension, and sample count. Incompatible legacy caches are preserved and a separate cache is created. Run `ddpm-validate --help` for single-GPU options.

## Repository layout

```text
ddpm/
  data/          Image discovery and preprocessing
  engine/        Training, sampling, and evaluation workflows
  evaluation/    FID feature extraction and statistics
  models/        Diffusion model and U-Net
  modules/       Attention and U-Net building blocks
  utils/         Checkpoints, EMA, distributed setup, and helpers
tools/           CLI wrappers, analysis tools, and benchmarks
tests/           Unit and distributed workflow tests
```

## Tests and benchmarks

Install the optional test dependencies as shown above, then run:

```bash
python -m pytest
```

Ubuntu CI runs the suite with two CPU/Gloo workers. The test below requires Linux, PyTorch NCCL support, and at least two CUDA GPUs; otherwise it is skipped:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/test_cuda_distributed.py -q
```

To compare the EMA and attention implementations and FID statistics on the target machine, run the benchmark without other GPU workloads:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m tools.benchmark_optimizations \
  --device cuda:0 --attention-batch 2 --fid-count 50000 --fid-dims 2048
```

Timings depend on hardware and software. Chunked FID statistics reduce peak memory and may take longer than `numpy.cov`.

## Troubleshooting

- **CUDA is unavailable:** check `nvidia-smi` and `torch.cuda.is_available()`, then install a PyTorch CUDA build compatible with your driver.
- **Invalid training batch size:** make the global `--batch-size` positive and divisible by the number of processes, and provide at least that many image files.
- **Resume configuration differs:** reuse the saved run's data path and training settings listed in the resume section.
- **Not enough real images for FID:** provide at least `--fid-samples` image files or lower the count for a smoke test. Scores with different counts are not directly comparable.

## References

- Ho, Jain, and Abbeel, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020.
- Song, Meng, and Ermon, [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), 2020.

## License

This project is distributed under the [Apache License 2.0](LICENSE). Individual Python file headers preserve their existing copyright notices.
