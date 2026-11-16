# DDPM

PyTorch implementation of denoising diffusion with a U-Net, DDPM and DDIM sampling, optional self-conditioning and learned variance, and FID evaluation.

## Installation

Training is intended for Ubuntu with NVIDIA GPUs. Install a CUDA-enabled PyTorch and matching torchvision for your driver using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install this project:

```bash
python -m pip install -e '.[test]'
```

The project requires Python 3.10+ and PyTorch 2.6+. The CPU device is available for smoke tests; production training defaults to CUDA. SDPA automatically selects a supported attention backend, including a fallback on GPUs without Flash Attention.

## Data

Point `--data` at a directory containing RGB images. Files are discovered recursively, so either a flat directory or class subdirectories work; labels are ignored. Training resizes to the configured resolution and normalizes to `[-1, 1]`. FID uses the same resize and crop with real and generated images in `[0, 1]` and no random augmentation.

## Multi-GPU training

On one Ubuntu machine, launch one process per visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli train --data ./datasets/cifar10/train \
  --batch-size 128 --save-dir ./results/cifar10
```

`--batch-size` is the **global microbatch** across all processes and must be divisible by the process count. The effective batch per optimizer step is `--batch-size * --accumulate`. A distributed sampler assigns distinct images to each rank; incomplete batches are dropped. The dataset must contain at least one global batch. The `cifar10` and `celeba` presets set default resolution and training parameters; select the latter with `--preset celeba`.

Useful options include `--steps`, `--lr`, `--workers`, `--image-size`, `--dim`, `--timesteps`, `--sampling-steps`, `--save-every`, `--self-condition`, and `--learned-variance`. `--flash` enables PyTorch SDPA with automatic backend selection; `--no-flash` uses the explicit attention implementation. See `python -m ddpm.cli train --help` for all options.

Samples and checkpoints are written under the save directory. The final step always produces `weights/last.pkl`, even when it falls between sampling intervals. Enable training-time FID with `--validate --fid-samples 50000 --fid-stat-path ./datasets/stats/train.npz`; at least that many distinct real images are required. The first run downloads the FID Inception weights. All ranks generate FID images using the rank-0 EMA model. FID still pauses training while features and the covariance square root are calculated; rank 0 prepares and caches the real-image statistics. `--fid-every 50000` sets a separate evaluation interval in optimizer steps, while the default is `--save-every`. Evaluations also save a checkpoint so the best FID weights are available.

## Resume, sample and evaluate

Resume with the same number of processes, dataset, global batch, augmentation, workers, accumulation, gradient clipping and seed. The checkpoint records the online and EMA weights, optimizer, AMP scaler, rank-specific RNG states and sampler position. New runs use reproducible per-image augmentation and skip directly to the saved sampler position without decoding earlier batches. Version 2 and older checkpoints retain their original augmentation and data replay behavior. `--steps` is the total target step, not the number of additional steps:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli train --data ./datasets/cifar10/train --batch-size 128 \
  --resume ./results/cifar10/weights/last.pkl --steps 900000

ddpm-sample --checkpoint ./results/cifar10/weights/last.pkl \
  --num-samples 100 --batch-size 25 --output ./results/samples

ddpm-validate --checkpoint ./results/cifar10/weights/last.pkl \
  --data ./datasets/cifar10/train --fid-samples 50000 \
  --fid-stat-path ./datasets/stats/cifar10.npz --output ./results/val
```

Sampling and validation use EMA weights. Both can also use all GPUs via `torchrun`; the exact requested sample count is divided between workers. The sampling and FID batch size targets the global batch: each worker uses `max(1, batch_size // number_of_workers)`, so choose a divisible batch size for the requested total:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli sample --checkpoint ./results/cifar10/weights/last.pkl \
  --num-samples 100 --batch-size 64 --output ./results/samples

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  -m ddpm.cli validate --checkpoint ./results/cifar10/weights/last.pkl \
  --data ./datasets/cifar10/train --batch-size 128 --fid-samples 50000 \
  --fid-stat-path ./datasets/stats/cifar10.npz --output ./results/val
```

FID defaults to 2048 Inception features; other `--feature-dims` values are useful for smoke tests but are **not comparable** to standard FID. Cache files include the dataset, image metadata, transform, feature dimension and sample count. Incompatible legacy caches are preserved and a separate cache is created. Checkpoint aliases (`last.pkl` and `best.pkl`) use hard links to their numbered checkpoint when the filesystem supports them, otherwise copies are used; do not edit checkpoint files in place. Legacy checkpoints can be loaded, but their previously quantized optimizer state and missing RNG state cannot be recovered. Load checkpoints only from trusted sources.

## Tests

```bash
python -m pytest
```

The Ubuntu CI tests include two CPU distributed workers and cover data sharding, gradient averaging, resume, distributed sampling, EMA synchronization and FID, as well as persistent data-loader workers and checkpoint compatibility. `tests/test_cuda_distributed.py` additionally runs the same training, mid-epoch resume, sampling and FID paths with two GPUs and NCCL when available; it is skipped on machines without them. Run it on the target Ubuntu machine with:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/test_cuda_distributed.py -q
```

To compare the EMA and attention implementations and FID statistics on that machine, run without other GPU workloads:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m tools.benchmark_optimizations \
  --device cuda:0 --attention-batch 2 --fid-count 50000 --fid-dims 2048
```

The benchmark prints median GPU timings and a numerical comparison for FID statistics. Results depend on the GPU, driver, batch size and CPU BLAS configuration; the chunked FID calculation reduces peak memory and may take slightly longer than `numpy.cov`.

## License

This project is distributed under the [Apache License 2.0](LICENSE). Individual Python file headers preserve the existing copyright notices.
