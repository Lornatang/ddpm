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
import weakref
import random

import pytest
import torch
from torch import nn

from ddpm.models.gaussian_diffusion import GaussianDiffusion
from ddpm.models.unet import UNet
from ddpm.modules.attend import Attend
from ddpm.utils.ops import save_yaml

from conftest import small_diffusion


class DummyModel(nn.Module):
    in_channels = out_dim = 3
    self_condition = False

    def __init__(self):
        """Initialize a lightweight prediction model and tensor-reference log."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.fixed = None
        self.refs = []
        self.alive = []

    def forward(self, x, t, condition=None):
        """Record the input and return either a fixed or weight-based prediction.

        Args:
            x (torch.Tensor): Noisy input image batch.
            t (torch.Tensor): Diffusion time steps, accepted for interface compatibility.
            condition (torch.Tensor | None, optional): Self-conditioning input. Defaults to ``None``.

        Returns:
            torch.Tensor: Fixed prediction when configured; otherwise the learnable weight broadcast over the input.
        """
        self.refs.append(weakref.ref(x))
        self.alive.append(sum(ref() is not None for ref in self.refs))
        return torch.zeros_like(x) + self.weight if self.fixed is None else self.fixed


def test_forward_diffusion_and_v_prediction_round_trip():
    diffusion = GaussianDiffusion(DummyModel(), 4, time_steps=32, sampling_time_steps=3)
    x = torch.randn(3, 3, 4, 4).clamp(-1, 1)
    noise = torch.randn_like(x)
    t = torch.tensor([0, 12, 31])
    noisy = diffusion.q_sample(x, t, noise)
    v = diffusion.prediction_latent_var(x, t, noise)
    recovered = diffusion.prediction_start_from_latent_var(noisy, t, v)
    predicted_noise, predicted_start = diffusion.model_predictions(noisy, t)
    assert predicted_noise.shape == predicted_start.shape == x.shape
    torch.testing.assert_close(recovered, x, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(diffusion.prediction_noise_from_start(noisy, t, recovered), noise, atol=1e-5, rtol=1e-5)
    mean, variance, _ = diffusion.q_posterior(x[:1], noisy[:1], t[:1])
    torch.testing.assert_close(mean, x[:1])
    assert variance.item() == 0


@pytest.mark.parametrize("steps", [3, 32])
def test_sampling_returns_finite_images_without_accumulating_history(steps):
    model = DummyModel()
    diffusion = GaussianDiffusion(model, 4, time_steps=32, sampling_time_steps=steps)
    result = diffusion.sample(batch_size=2)
    assert result.shape == (2, 3, 4, 4)
    assert torch.isfinite(result).all() and not result.requires_grad
    assert result.min() >= 0 and result.max() <= 1
    assert max(model.alive) <= 2
    trajectory = diffusion.sample(batch_size=2, return_all_time_steps=True)
    assert trajectory.shape == (2, steps + 1, 3, 4, 4)


def test_immiscible_assignment_uses_same_noise_for_loss():
    model = DummyModel()
    diffusion = GaussianDiffusion(model, 4, time_steps=32, sampling_time_steps=3, immiscible=True)
    images = torch.stack([torch.ones(3, 4, 4), -torch.ones(3, 4, 4)])
    noise = -images
    t = torch.tensor([15, 15])
    assignment = diffusion.noise_assignment(images, noise)
    assert assignment.tolist() == [1, 0]
    model.fixed = diffusion.prediction_latent_var(images, t, noise[assignment])
    torch.testing.assert_close(diffusion.p_losses(images, t, noise), torch.tensor(0.0))


def test_unet_options_and_learned_variance_receive_gradients(monkeypatch):
    unet = UNet(8, init_dim=16, out_dim=5, channels_multipliers=(1, 2), attn_heads=1, attn_dim_head=4)
    assert unet(torch.randn(2, 3, 8, 8), torch.tensor([0, 1])).shape == (2, 5, 8, 8)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    conditioned = small_diffusion(self_condition=True)
    assert torch.isfinite(conditioned(torch.randn(2, 3, 8, 8)))
    combined = small_diffusion(self_condition=True, learned_variance=True)
    assert torch.isfinite(combined(torch.randn(2, 3, 8, 8).clamp(-1, 1)))

    learned = small_diffusion(learned_variance=True)
    x = torch.rand(2, 3, 8, 8) * 2 - 1
    loss = learned.p_losses(x, torch.tensor([0, 15]))
    assert torch.isfinite(loss)
    loss.backward()
    grad = learned.model.final_conv.weight.grad
    assert torch.isfinite(grad).all() and grad[:3].abs().sum() > 0 and grad[3:].abs().sum() > 0
    mean, variance, log_variance, _ = learned.p_mean_variance(x, torch.tensor([0, 15]))
    assert mean.shape == variance.shape == log_variance.shape == x.shape
    assert torch.isfinite(variance).all() and (variance > 0).all()
    learned.eval()
    learned.is_ddim_sampling = False
    assert torch.isfinite(learned.sample(batch_size=1)).all()


def test_interpolation_and_invalid_sampling_schedules():
    model = DummyModel()
    diffusion = GaussianDiffusion(model, 4, time_steps=32, sampling_time_steps=3)
    seen = []
    diffusion.p_sample = lambda x, t, cond: (seen.append(t) or x, None)
    images = torch.zeros(1, 3, 4, 4)
    diffusion.interpolate(images, images, time_step=3)
    assert seen == [3, 2, 1, 0]
    seen.clear()
    diffusion.interpolate(images, images)
    assert seen == list(reversed(range(32)))
    with pytest.raises(ValueError, match="time_steps"):
        GaussianDiffusion(model, 4, time_steps=20, sampling_time_steps=3)
    with pytest.raises(ValueError, match="sampling_time_steps"):
        GaussianDiffusion(model, 4, time_steps=32, sampling_time_steps=0)


def test_sdpa_automatically_falls_back_and_preserves_scale():
    q = torch.randn(1, 2, 4, 8, requires_grad=True)
    k = torch.randn(1, 2, 5, 8, requires_grad=True)
    v = torch.randn(1, 2, 5, 8, requires_grad=True)
    fast = Attend(flash=True, scale=0.13)(q, k, v)
    reference = Attend(flash=False, scale=0.13)(q, k, v)
    torch.testing.assert_close(fast, reference, atol=1e-6, rtol=1e-5)
    assert fast.dtype == q.dtype
    fast.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_yaml_serialization_does_not_modify_source(tmp_path):
    config = {"device": torch.device("cpu"), "root": tmp_path}
    save_yaml(tmp_path / "config.yaml", config)
    assert isinstance(config["device"], torch.device)
    assert config["root"] == tmp_path
