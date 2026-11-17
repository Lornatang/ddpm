# Copyright (c) Lornatang. All rights reserved.
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
"""Compare the old and current EMA, attention, and FID statistics paths.

Run on the target GPU without other GPU workloads. Timings are informational;
they are deliberately not CI thresholds, as hardware and drivers vary.
"""

import argparse
import copy
import statistics
import time

import numpy as np
import torch
from einops import rearrange, repeat

from ddpm.evaluation.fid_metrics import feature_statistics
from ddpm.models.gaussian_diffusion import GaussianDiffusion
from ddpm.models.unet import UNet
from ddpm.modules.attention import Attention
from ddpm.utils.ema import ModelEMA


def measure(function, warmup, repeats, device):
    """Measure a synchronized CUDA workload and report its median latency.

    Args:
        function (Callable): Zero-argument workload to measure.
        warmup (int): Number of untimed warmup iterations.
        repeats (int): Number of timed iterations.
        device (torch.device): CUDA device to synchronize.

    Returns:
        float: Median elapsed time per iteration in milliseconds.
    """
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000)
    return statistics.median(elapsed)


def benchmark_ema(device, warmup, repeats):
    """Print scalar and grouped EMA update timings for a representative model.

    Args:
        device (torch.device): CUDA device used for the benchmark.
        warmup (int): Number of untimed iterations per implementation.
        repeats (int): Number of timed iterations per implementation.
    """
    model = GaussianDiffusion(UNet(128, channels_multipliers=(1, 2, 2, 2), flash=True), 32, 1000,
                              sampling_time_steps=250).to(device)
    ema = ModelEMA(model)
    scalar = copy.deepcopy(ema.ema).state_dict()
    source = model.state_dict()

    @torch.no_grad()
    def previous():
        """Apply the earlier per-tensor EMA update to the benchmark state."""
        for key, value in scalar.items():
            if value.is_floating_point():
                value *= 0.999
                value += 0.001 * source[key].detach()

    old = measure(previous, warmup, repeats, device)
    new = measure(lambda: ema.update(model), warmup, repeats, device)
    print(f"EMA median: scalar={old:.3f} ms, foreach={new:.3f} ms, speedup={old / new:.2f}x")


def old_attention(layer, x):
    """Run attention using the previous query and memory-token layout.

    Args:
        layer (Attention): Attention module whose weights are used.
        x (torch.Tensor): Input feature map.

    Returns:
        torch.Tensor: Attention output using the previous layout.
    """
    batch, _, height, width = x.shape
    qkv = layer.to_qkv(layer.norm(x)).chunk(3, dim=1)
    q, k, v = (rearrange(t, "b (h c) x y -> b h (x y) c", h=layer.heads) for t in qkv)
    mk, mv = (repeat(t, "h n d -> b h n d", b=batch) for t in layer.mem_kv)
    k, v = torch.cat((mk, k), dim=-2), torch.cat((mv, v), dim=-2)
    output = layer.attend(q, k, v)
    return layer.to_out(rearrange(output, "b h (x y) d -> b (h d) x y", x=height, y=width))


def benchmark_attention(device, warmup, repeats, batch):
    """Print forward/backward timings for the previous and current attention layouts.

    Args:
        device (torch.device): CUDA device used for the benchmark.
        warmup (int): Number of untimed iterations per implementation.
        repeats (int): Number of timed iterations per implementation.
        batch (int): Batch size for the attention input.
    """
    layer = Attention(256, heads=4, dim_head=32, flash=True).to(device).train()
    x = torch.randn(batch, 256, 32, 32, device=device, requires_grad=True)

    def run(previous):
        """Run one attention forward and backward pass.

        Args:
            previous (bool): Use the previous attention layout when ``True``.
        """
        layer.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast("cuda", dtype=torch.float16):
            output = old_attention(layer, x) if previous else layer(x)
            loss = output.float().square().mean()
        loss.backward()

    old = measure(lambda: run(True), warmup, repeats, device)
    new = measure(lambda: run(False), warmup, repeats, device)
    print(f"Attention forward/backward median: old={old:.3f} ms, current={new:.3f} ms, speedup={old / new:.2f}x")


def benchmark_fid(count, dims):
    """Compare NumPy and chunked FID statistics timings and numerical results.

    Args:
        count (int): Number of synthetic feature vectors.
        dims (int): Number of dimensions per feature vector.
    """
    features = np.random.default_rng(0).standard_normal((count, dims), dtype=np.float32)
    start = time.perf_counter()
    old_mean = features.mean(axis=0, dtype=np.float64)
    old_covariance = np.cov(features, rowvar=False)
    old = time.perf_counter() - start
    start = time.perf_counter()
    new_mean, new_covariance = feature_statistics(features)
    new = time.perf_counter() - start
    print(f"FID statistics: numpy={old:.3f} s, chunked={new:.3f} s, "
          f"max mean error={np.max(np.abs(old_mean - new_mean)):.3g}, "
          f"max covariance error={np.max(np.abs(old_covariance - new_covariance)):.3g}")


def main():
    """Parse benchmark options and run the CUDA and FID comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--attention-batch", type=int, default=2)
    parser.add_argument("--fid-count", type=int, default=5000)
    parser.add_argument("--fid-dims", type=int, default=256)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.attention_batch < 1 or args.fid_count < 2 or args.fid_dims < 1:
        parser.error("benchmark sizes and repeat counts must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("a CUDA device is required for the GPU benchmarks")
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}; PyTorch: {torch.__version__}")
    benchmark_ema(device, args.warmup, args.repeats)
    benchmark_attention(device, args.warmup, args.repeats, args.attention_batch)
    benchmark_fid(args.fid_count, args.fid_dims)


if __name__ == "__main__":
    main()
