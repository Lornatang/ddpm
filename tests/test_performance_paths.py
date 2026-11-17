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
import copy
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import nn

from ddpm.data.base import BaseImageFolder
from ddpm.evaluation.fid_metrics import feature_statistics
from ddpm.modules.attention import Attention
from ddpm.utils.ema import ModelEMA
from test_workflow import assert_state_equal, make_trainer


class MixedModel(nn.Module):
    def __init__(self):
        """Create parameters and buffers with mixed floating-point and integer dtypes."""
        super().__init__()
        self.p = nn.Parameter(torch.tensor([1.0, -2.0]))
        self.double_param = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float64))
        self.register_buffer("float_buffer", torch.tensor([12.0]))
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))


def test_foreach_ema_matches_scalar_reference_and_keeps_integer_buffers():
    model = MixedModel()
    ema = ModelEMA(model, decay=0.99, tau=5)
    reference = copy.deepcopy(ema.ema.state_dict())
    with torch.no_grad():
        for step in range(30):
            model.p.add_(0.01 * step)
            model.double_param.sub_(0.02 * step)
            model.float_buffer.add_(0.1)
            model.counter.add_(1)
            decay = ema.decay(step + 1)
            for key, value in reference.items():
                if value.is_floating_point():
                    value *= decay
                    value += (1 - decay) * model.state_dict()[key]
            ema.update(model)
    for name, value in ema.ema.state_dict().items():
        torch.testing.assert_close(value, reference[name], rtol=2e-6, atol=2e-6)
    assert ema.updates == 30 and ema.ema.counter.item() == 7
    ema.enabled = False
    ema.update(model)
    assert ema.updates == 30
    assert all(parameter.grad is None for parameter in ema.ema.parameters())


def legacy_attention(layer, x):
    """Run the attention layout used before the SDPA optimization.

    Args:
        layer (Attention): Attention module providing the weights.
        x (torch.Tensor): Input feature map.

    Returns:
        torch.Tensor: Attention output using the previous tensor layout.
    """
    b, _, h, w = x.shape
    qkv = layer.to_qkv(layer.norm(x)).chunk(3, dim=1)
    q, k, v = (rearrange(t, "b (h c) x y -> b h (x y) c", h=layer.heads) for t in qkv)
    mk, mv = (repeat(t, "h n d -> b h n d", b=b) for t in layer.mem_kv)
    k, v = torch.cat((mk, k), dim=-2), torch.cat((mv, v), dim=-2)
    out = layer.attend(q, k, v)
    return layer.to_out(rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w))


@pytest.mark.parametrize("autocast", [False, True])
def test_attention_preserves_outputs_and_gradients_with_fused_layout(autocast):
    torch.manual_seed(4)
    current = Attention(16, heads=2, dim_head=4, flash=True)
    old = copy.deepcopy(current)
    x = torch.randn(2, 16, 4, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    seen = []
    hook = current.attend.register_forward_pre_hook(lambda module, inputs: seen.append(inputs))
    with torch.autocast("cpu", enabled=autocast, dtype=torch.bfloat16):
        actual = current(x)
        expected = legacy_attention(old, y)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    hook.remove()
    q, k, v = seen[0]
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    assert q.dtype == k.dtype == v.dtype
    tolerance = 2e-2 if autocast else 1e-5
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x.grad, y.grad, rtol=tolerance, atol=tolerance)
    for left, right in zip(current.parameters(), old.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=tolerance, atol=tolerance)


def test_chunked_fid_statistics_match_numpy_with_large_offset():
    rng = np.random.default_rng(33)
    features = (rng.normal(size=(103, 9)) + 100000).astype(np.float32)
    mean, covariance = feature_statistics(features, chunk_size=7)
    np.testing.assert_allclose(mean, features.mean(axis=0, dtype=np.float64), rtol=0, atol=1e-9)
    np.testing.assert_allclose(covariance, np.cov(features, rowvar=False), rtol=1e-10, atol=1e-10)


def test_mid_epoch_resume_reads_only_the_next_batch(image_dir, tmp_path, monkeypatch):
    trainer = make_trainer(image_dir, tmp_path / "run", steps=3)
    trainer.train()
    checkpoint = tmp_path / "run" / "weights" / "last.pkl"
    resumed = make_trainer(image_dir, tmp_path / "unused", steps=4, resume=checkpoint)
    original = BaseImageFolder.__getitem__
    reads = []

    def counted(dataset, index):
        reads.append(index)
        return original(dataset, index)

    monkeypatch.setattr(BaseImageFolder, "__getitem__", counted)
    resumed._setup_train()
    assert resumed.batches_in_epoch == 3
    resumed._next_batch()
    assert len(reads) == 2


def test_legacy_data_replay_remains_exact(image_dir, tmp_path):
    def legacy_trainer(output, steps):
        trainer = make_trainer(image_dir, output, steps=steps)
        trainer._resume_checkpoint = {"step": 0, "rng_states": [trainer._rng_state()]}
        return trainer

    continuous = legacy_trainer(tmp_path / "continuous", 3)
    continuous.train()
    split = legacy_trainer(tmp_path / "split", 1)
    split.train()
    path = tmp_path / "split" / "weights" / "last.pkl"
    saved = torch.load(path, weights_only=False)
    assert saved["data_mode"] == "legacy"
    saved.pop("data_mode")
    saved["format_version"] = 2
    legacy_path = path.with_name("legacy.pkl")
    torch.save(saved, legacy_path)
    resumed = make_trainer(image_dir, tmp_path / "unused", steps=3, resume=legacy_path)
    resumed.train()
    assert resumed.data_mode == "legacy"
    assert_state_equal(continuous.model.state_dict(), resumed.model.state_dict())
    assert_state_equal(continuous.ema.ema.state_dict(), resumed.ema.ema.state_dict())


def test_cli_reads_resume_checkpoint_once(image_dir, tmp_path):
    from ddpm.cli import train_main

    trainer = make_trainer(image_dir, tmp_path / "run", steps=1)
    trainer.train()
    checkpoint = tmp_path / "run" / "weights" / "last.pkl"
    original = torch.load
    with patch("torch.load", wraps=original) as load:
        train_main(["--data", str(image_dir), "--resume", str(checkpoint), "--batch-size", "2",
                    "--workers", "0", "--device", "cpu", "--steps", "2", "--seed", "17",
                    "--save-every", "100", "--augment"])
    assert load.call_count == 1


def test_checkpoint_aliases_and_copy_fallback(image_dir, tmp_path, monkeypatch):
    original_link = os.link
    link_supported = [True]

    def try_link(*args, **kwargs):
        try:
            return original_link(*args, **kwargs)
        except OSError:
            link_supported[0] = False
            raise

    monkeypatch.setattr("ddpm.engine.trainer.os.link", try_link)
    trainer = make_trainer(image_dir, tmp_path / "linked", steps=1)
    trainer.train()
    weights = tmp_path / "linked" / "weights"
    assert (weights / "last.pkl").samefile(weights / "1.pkl") == link_supported[0]

    def unsupported(*args, **kwargs):
        raise OSError("hard links unavailable")

    monkeypatch.setattr("ddpm.engine.trainer.os.link", unsupported)
    other = make_trainer(image_dir, tmp_path / "copied", steps=1)
    other.train()
    weights = tmp_path / "copied" / "weights"
    assert not (weights / "last.pkl").samefile(weights / "1.pkl")
    assert (weights / "last.pkl").read_bytes() == (weights / "1.pkl").read_bytes()


def test_persistent_worker_augmentation_resumes_exactly(image_dir, tmp_path):
    from ddpm.data.base import ReproducibleImageFolder, image_transform
    from ddpm.engine.trainer import Trainer
    from test_workflow import build_model

    for path in image_dir.glob("*.png"):
        pixels = np.asarray(Image.open(path)).copy()
        pixels[:, :4, 0] = 0
        pixels[:, 4:, 0] = 255
        Image.fromarray(pixels).save(path)
    dataset = ReproducibleImageFolder(image_dir, image_transform((8, 8)), augment=True, seed=17)
    assert any(not torch.equal(dataset[(i, 0)], dataset[(i, 1)]) for i in range(len(dataset)))
    torch.testing.assert_close(dataset[(0, 4)], dataset[(0, 4)], rtol=0, atol=0)

    def make(output, steps, resume=None):
        return Trainer(build_model(), image_dir, augment=True, resume=resume, batch_size=2, num_workers=2,
                       num_steps=steps, save_and_sample_every=100, num_samples=1, device="cpu",
                       save_dir=output, seed=17)

    continuous = make(tmp_path / "continuous", 6)
    continuous.train()
    split = make(tmp_path / "split", 2)
    split.train()
    checkpoint = tmp_path / "split" / "weights" / "last.pkl"
    resumed = make(tmp_path / "unused", 6, resume=checkpoint)
    resumed.train()
    assert resumed.train_loader.persistent_workers
    assert_state_equal(continuous.model.state_dict(), resumed.model.state_dict())
    assert_state_equal(continuous.ema.ema.state_dict(), resumed.ema.ema.state_dict())


def test_fid_interval_is_independent_of_sampling(image_dir, tmp_path, monkeypatch):
    import ddpm.engine.trainer as trainer_module

    calls = []

    class FakeFID:
        def __init__(self, *args, **kwargs):
            pass

        def calculate(self):
            calls.append(1)
            return 3.0 - len(calls)

    monkeypatch.setattr(trainer_module, "FIDMetrics", FakeFID)
    trainer = make_trainer(image_dir, tmp_path / "run", steps=6, val=True, save_every=2)
    trainer.fid_every = 3
    trainer.train()
    assert len(calls) == 2
    assert sorted(path.name for path in (tmp_path / "run" / "samples").glob("*.png")) == ["2.png", "4.png", "6.png"]
    best = torch.load(tmp_path / "run" / "weights" / "best.pkl", weights_only=False)
    assert best["step"] == 6 and best["best_fid_score"] == 1.0


def test_single_gpu_inference_respects_visible_device_mapping(monkeypatch):
    from ddpm.utils.distributed import DistributedInference

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    context = DistributedInference("cuda:1")
    assert context.device == torch.device("cuda:1")
    assert selected == [torch.device("cuda:1")]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,3"
