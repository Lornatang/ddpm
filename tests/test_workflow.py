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
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import ddpm.engine.trainer as trainer_module
import ddpm.evaluation.fid_metrics as fid_module
from ddpm.data.base import BaseImageFolder, image_transform
from ddpm.engine.inference import Inferencer
from ddpm.engine.trainer import Trainer
from ddpm.engine.validator import Validator
from ddpm.evaluation.fid_metrics import FIDMetrics, frechet_distance
from ddpm.utils.checkpoint import load_checkpoint, model_config

from conftest import small_diffusion


def build_model():
    """Build a reproducibly initialized small diffusion model.

    Returns:
        GaussianDiffusion: Model used by the workflow tests.
    """
    torch.manual_seed(123)
    return small_diffusion()


def make_trainer(root, output, *, steps, resume=None, val=False, save_every=100):
    """Build a CPU trainer with the workflow tests' shared settings.

    Args:
        root (Path | str): Directory containing training images.
        output (Path | str): Directory for a new run's outputs.
        steps (int): Total number of optimizer steps.
        resume (Path | str | None, optional): Checkpoint to resume. Defaults to ``None``.
        val (bool, optional): Enable FID evaluation. Defaults to ``False``.
        save_every (int, optional): Checkpoint and preview interval. Defaults to 100.

    Returns:
        Trainer: Configured CPU trainer.
    """
    return Trainer(build_model(), root, augment=True, resume=resume, batch_size=2,
                   num_workers=0, num_steps=steps, save_and_sample_every=save_every,
                   num_samples=2, val=val, num_fid_samples=4, device="cpu",
                   save_dir=output, seed=17)


def assert_state_equal(first, second):
    """Assert two state dictionaries have identical keys and tensor values.

    Args:
        first (dict): First state dictionary.
        second (dict): State dictionary to compare against the first.
    """
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0, msg=name)


def test_training_checkpoint_resume_matches_uninterrupted_run(image_dir, tmp_path):
    uninterrupted = make_trainer(image_dir, tmp_path / "continuous", steps=3)
    uninterrupted.train()
    split = make_trainer(image_dir, tmp_path / "resumed", steps=1)
    split.train()
    checkpoint_path = tmp_path / "resumed" / "weights" / "last.pkl"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["step"] == 1 and saved["batches_in_epoch"] == 1
    assert all(v.dtype == torch.float32 for v in saved["model"].values() if v.is_floating_point())
    moments = [v["exp_avg_sq"] for v in saved["optimizer"]["state"].values() if "exp_avg_sq" in v]
    assert moments and all(v.dtype == torch.float32 for v in moments)

    resumed = make_trainer(image_dir, tmp_path / "unused", steps=3, resume=checkpoint_path)
    resumed.train()
    assert_state_equal(uninterrupted.model.state_dict(), resumed.model.state_dict())
    assert_state_equal(uninterrupted.ema.ema.state_dict(), resumed.ema.ema.state_dict())
    for left, right in zip(uninterrupted.optimizer.state.values(), resumed.optimizer.state.values()):
        assert_state_equal(left, right)
    assert resumed.current_step == 3

    loaded, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    assert checkpoint["step"] == 3
    assert_state_equal(loaded.state_dict(), resumed.ema.ema.state_dict())
    output = tmp_path / "generated"
    Inferencer(checkpoint_path, batch_size=2, num_samples=3, save_dir=output, device="cpu").inference()
    assert len(list(output.glob("*.png"))) == 3

    mismatched = Trainer(build_model(), image_dir, augment=True, resume=checkpoint_path, batch_size=4,
                         num_workers=0, num_steps=4, save_and_sample_every=100, num_samples=2,
                         device="cpu", seed=17)
    with pytest.raises(ValueError, match="batch_size"):
        mismatched.train()

    script = Path(__file__).resolve().parents[1] / "tools" / "analysis" / "reverse_diffusion_visualizer.py"
    spec = importlib.util.spec_from_file_location("reverse_visualizer", script)
    visualizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visualizer)
    images = visualizer.reverse_diffusion(loaded, torch.randn(1, 3, 8, 8), 32, "cpu")
    assert len(images) == 1 and images[0].shape == (8, 8, 3)


def test_best_fid_is_saved_before_checkpoint(image_dir, tmp_path, monkeypatch):
    class FakeFID:
        def __init__(self, *args, **kwargs):
            self.scores = iter([1.0, 2.0])

        def calculate(self):
            return next(self.scores)

    monkeypatch.setattr(trainer_module, "FIDMetrics", FakeFID)
    trainer = make_trainer(image_dir, tmp_path / "best", steps=2, val=True, save_every=1)
    trainer.train()
    last = torch.load(tmp_path / "best" / "weights" / "last.pkl", weights_only=False)
    best = torch.load(tmp_path / "best" / "weights" / "best.pkl", weights_only=False)
    assert last["step"] == 2 and last["best_fid_score"] == 1.0
    assert best["step"] == 1 and best["best_fid_score"] == 1.0


class SpyInception(nn.Module):
    """Record image batches and expose channel means as test features."""
    BLOCK_INDEX_BY_DIM = {64: 0, 192: 1, 768: 2, 2048: 3}
    instances = []

    def __init__(self, blocks):
        """Initialize the spy with an Inception-compatible signature.

        Args:
            blocks (list[int]): Requested Inception blocks; accepted for interface compatibility.
        """
        super().__init__()
        self.batches = []
        self.instances.append(self)

    def forward(self, samples):
        """Record images and return their spatial channel means.

        Args:
            samples (torch.Tensor): RGB image batch.

        Returns:
            list[torch.Tensor]: One feature-map batch with a channel per input channel.
        """
        self.batches.append(samples.detach().cpu().clone())
        return [samples.mean(dim=(2, 3), keepdim=True)]


class ReplayModel(nn.Module):
    """Return predetermined samples in order for FID tests."""

    def __init__(self, samples):
        """Initialize the sample sequence.

        Args:
            samples (torch.Tensor): Image batch to replay.
        """
        super().__init__()
        self.samples, self.offset = samples, 0

    def sample(self, batch_size):
        """Return the next slice of the stored samples.

        Args:
            batch_size (int): Number of samples to request.

        Returns:
            torch.Tensor: Next slice of the stored image batch.
        """
        out = self.samples[self.offset:self.offset + batch_size]
        self.offset += batch_size
        return out


def test_fid_uses_exact_counts_ranges_and_preserves_old_cache(image_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(fid_module, "InceptionV3", SpyInception)
    dataset = BaseImageFolder(image_dir, image_transform((8, 8)))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    samples = torch.stack([dataset[i] for i in range(5)])
    old_cache = tmp_path / "fid.npz"
    np.savez(old_cache, m2=np.zeros(3), s2=np.eye(3))
    previous_bytes = old_cache.read_bytes()

    metrics = FIDMetrics(loader, batch_size=3, model=ReplayModel(samples), dataset_stat_path=old_cache,
                         device="cpu", num_samples=5, feature_dims=64)
    score = metrics.calculate()
    assert abs(score) < 1e-6
    batches = metrics.inception_v3.batches
    assert [len(batch) for batch in batches] == [2, 2, 1, 3, 2]
    assert all(batch.min() >= 0 and batch.max() <= 1 for batch in batches)
    assert old_cache.read_bytes() == previous_bytes
    assert len(list(tmp_path.glob("fid-*.npz"))) == 1

    cached = FIDMetrics(loader, 3, ReplayModel(samples), old_cache, device="cpu", num_samples=5, feature_dims=64)
    cached.load_dataset_stat()
    assert cached.inception_v3.batches == []
    with pytest.raises(ValueError, match="distinct real images"):
        FIDMetrics(loader, 3, ReplayModel(samples), old_cache, device="cpu", num_samples=9, feature_dims=64)


def test_frechet_distance_for_diagonal_gaussians():
    mean1, mean2 = np.array([0.0, 2.0]), np.array([1.0, 0.0])
    covariance1, covariance2 = np.diag([1.0, 4.0]), np.diag([9.0, 1.0])
    assert frechet_distance(mean1, covariance1, mean2, covariance2) == pytest.approx(10.0)


def test_validation_workflow_loads_checkpoint_and_computes_fid(image_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(fid_module, "InceptionV3", SpyInception)
    trainer = make_trainer(image_dir, tmp_path / "train", steps=1)
    trainer.train()
    checkpoint = tmp_path / "train" / "weights" / "last.pkl"
    validator = Validator(checkpoint, image_dir, batch_size=2, num_fid_samples=4,
                          fid_stat_path=tmp_path / "stats.npz", save_dir=tmp_path / "val",
                          device="cpu", num_workers=0, feature_dims=64)
    score = validator.validate()
    assert np.isfinite(score)
    assert len(list((tmp_path / "val").glob("*.png"))) == 2


def test_legacy_module_checkpoint_still_loads(tmp_path):
    model = build_model()
    expected = copy.deepcopy(model).half().float().state_dict()
    del model.config
    del model.learned_variance
    del model.model.config
    path = tmp_path / "legacy.pkl"
    torch.save({"model": None, "ema": model.half(), "step": 1}, path)
    loaded, _ = load_checkpoint(path, "cpu")
    assert_state_equal(loaded.state_dict(), expected)
    assert model_config(loaded)["unet"]["channels_multipliers"] == (1, 2)
    assert torch.isfinite(loaded.sample(batch_size=1)).all()
