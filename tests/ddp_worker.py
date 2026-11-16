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
"""Two-process CPU worker used to exercise the real torchrun training path."""

import os
import sys
from pathlib import Path

import torch
from torch import distributed as dist, nn

from ddpm.engine.trainer import Trainer
from ddpm.models.gaussian_diffusion import GaussianDiffusion
from ddpm.models.unet import UNet


class ToyDiffusion(nn.Module):
    image_size = (8, 8)

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.batches = []

    def forward(self, x):
        factor = torch.rand((), device=x.device)
        self.batches.append((x.detach().clone(), factor.detach()))
        return (self.weight * x).square().mean() * factor


def main():
    torch.set_num_threads(1)
    # Some local development builds ship torch without its distribution metadata,
    # so the new lazy backend registration entry point is unavailable.
    if os.name == "nt":
        from torch.distributed import distributed_c10d
        if hasattr(distributed_c10d, "_register_builtin_gloo_backend"):
            distributed_c10d._register_builtin_gloo_backend()
    root, output = map(Path, sys.argv[1:])
    steps = int(os.environ.get("DDPM_TEST_STEPS", "2"))
    resume = os.environ.get("DDPM_TEST_RESUME")
    if os.environ.get("DDPM_TEST_MODEL") == "unet":
        torch.manual_seed(19)
        model = GaussianDiffusion(UNet(8, channels_multipliers=(1, 2), attn_heads=1,
                                       attn_dim_head=4, flash=True), 8, 32, sampling_time_steps=3)
    else:
        model = ToyDiffusion()
    trainer = Trainer(model, root, resume=resume, batch_size=4, num_workers=0, num_steps=steps,
                      gradient_accumulate_every=2, save_and_sample_every=100,
                      num_samples=1, device="cpu", save_dir=output, seed=23)
    setup = trainer._setup_train
    update = trainer.optimizer_step

    def checked_setup():
        setup()
        indices = list(trainer.train_sampler)
        all_indices = [None] * trainer.world_size
        dist.all_gather_object(all_indices, indices)
        assert len(all_indices) == 2
        assert set(all_indices[0]).isdisjoint(all_indices[1])
        assert len(set(all_indices[0]) | set(all_indices[1])) == 8

    def checked_update():
        model = trainer.model.module
        if isinstance(model, ToyDiffusion):
            local_gradient = sum(2 * model.weight.detach() * x.square().mean() * factor
                                 for x, factor in model.batches) / len(model.batches)
            dist.all_reduce(local_gradient)
            local_gradient /= trainer.world_size
            torch.testing.assert_close(model.weight.grad, local_gradient)
            model.batches.clear()
        update()
        weights = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        replicas = [torch.zeros_like(weights) for _ in range(trainer.world_size)]
        dist.all_gather(replicas, weights)
        torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)

    trainer._setup_train = checked_setup
    trainer.optimizer_step = checked_update
    trainer.train()
    (output / f"rank-{os.environ['RANK']}.ok").write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
