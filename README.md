# ddpm

PyTorch implements `Denoising Diffusion Probabilistic Models` paper.

## DDP training

```bash
OMP_NUM_THREADS=1 torchrun --nproc-per-node=2 --master_port=16800 ./tools/train.py
```
