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

import ddpm.evaluation.fid_metrics as fid_module
from ddpm.engine.trainer import Trainer
from ddpm.engine.inference import Inferencer
from ddpm.engine.validator import Validator
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


class TestInception(nn.Module):
    BLOCK_INDEX_BY_DIM = {64: 0, 192: 1, 768: 2, 2048: 3}

    def __init__(self, blocks):
        super().__init__()
        self.batches = []

    def forward(self, samples):
        self.batches.append(len(samples))
        values = samples.mean(dim=(2, 3))
        values = torch.cat((values, torch.zeros(len(samples), 61, device=samples.device)), dim=1)
        return [values[:, :, None, None]]


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
    device = os.environ.get("DDPM_TEST_DEVICE", "cpu")
    validate = os.environ.get("DDPM_TEST_FID") == "1"
    if validate:
        fid_module.InceptionV3 = TestInception
        original_calculate = fid_module.FIDMetrics.calculate_distributed

        def checked_calculate(metric):
            values = torch.cat([value.detach().reshape(-1) for value in metric.model.state_dict().values()])
            replicas = [torch.empty_like(values) for _ in range(dist.get_world_size())]
            dist.all_gather(replicas, values)
            torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
            generated = []
            original_sample = metric.model.sample

            def sample(*, batch_size):
                images = original_sample(batch_size=batch_size)
                generated.append(images.cpu())
                return images

            metric.model.sample = sample
            try:
                score = original_calculate(metric)
            finally:
                del metric.model.sample
            count = metric.num_samples // dist.get_world_size() + (dist.get_rank() < metric.num_samples % dist.get_world_size())
            assert sum(map(len, generated)) == count
            samples = torch.cat(generated) if generated else torch.empty((0, 3, 8, 8))
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, samples)
            if dist.get_rank() == 0:
                features = metric.calculate_inception_features(torch.cat(gathered)).numpy()
                mean, covariance = fid_module.feature_statistics(features)
                reference = fid_module.frechet_distance(mean, covariance, metric.m2, metric.s2)
                assert abs(score - reference) < 1e-5
            return score

        fid_module.FIDMetrics.calculate_distributed = checked_calculate
    mode = os.environ.get("DDPM_TEST_MODE", "train")
    if mode == "sample":
        checkpoint = output / "weights" / "last.pkl"
        Inferencer(checkpoint, batch_size=4, num_samples=5, save_dir=output / "inference", device=device).inference()
        return
    if mode == "validate":
        checkpoint = output / "weights" / "last.pkl"
        validator = Validator(checkpoint, root, batch_size=4, num_fid_samples=5,
                              fid_stat_path=output / "stats.npz", save_dir=output / "validation",
                              device=device, num_workers=0, feature_dims=64)
        assert torch.isfinite(torch.tensor(validator.validate()))
        return
    if os.environ.get("DDPM_TEST_MODEL") == "unet":
        torch.manual_seed(19)
        model = GaussianDiffusion(UNet(8, channels_multipliers=(1, 2), attn_heads=1,
                                       attn_dim_head=4, flash=True), 8, 32, sampling_time_steps=3)
    else:
        model = ToyDiffusion()
    trainer = Trainer(model, root, resume=resume, batch_size=4, num_workers=0, num_steps=steps,
                      gradient_accumulate_every=1 if validate else 2,
                      save_and_sample_every=1 if validate else 100,
                      num_samples=1, device=device, save_dir=output, seed=23,
                      val=validate, num_fid_samples=5, fid_feature_dims=64,
                      fid_stat_path=output / "stats.npz", augment=validate)
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
    if validate:
        assert torch.isfinite(torch.tensor(trainer.fid_score))
        assert trainer.fid_metrics.inception_v3.batches
    (output / f"rank-{os.environ['RANK']}.ok").write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
