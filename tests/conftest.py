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
from pathlib import Path

import pytest
import torch
from PIL import Image

from ddpm.models.gaussian_diffusion import GaussianDiffusion
from ddpm.models.unet import UNet


def pytest_sessionstart(session):
    """Limit PyTorch CPU threading for the test session.

    Args:
        session (pytest.Session): Session supplied by the pytest hook.
    """
    torch.set_num_threads(1)


@pytest.fixture
def image_dir(tmp_path: Path):
    """Create a directory containing eight small RGB images.

    Args:
        tmp_path (Path): Per-test temporary directory supplied by pytest.

    Returns:
        Path: Directory containing the test images.
    """
    root = tmp_path / "images"
    root.mkdir()
    for i in range(8):
        Image.new("RGB", (8, 8), (i * 25, 255 - i * 20, i * 13)).save(root / f"{i}.png")
    return root


def small_diffusion(*, self_condition=False, learned_variance=False):
    """Build a small diffusion model for CPU and distributed smoke tests.

    Args:
        self_condition (bool, optional): Enable U-Net self-conditioning. Defaults to ``False``.
        learned_variance (bool, optional): Enable learned variance output. Defaults to ``False``.

    Returns:
        GaussianDiffusion: Model configured for 8-by-8 RGB images.
    """
    model = UNet(8, channels_multipliers=(1, 2), attn_heads=1, attn_dim_head=4,
                 self_condition=self_condition, learned_variance=learned_variance, dropout=0.1)
    return GaussianDiffusion(model, 8, time_steps=32, sampling_time_steps=3)
