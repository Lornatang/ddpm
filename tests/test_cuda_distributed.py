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
"""End-to-end NCCL coverage, automatically enabled on a two-GPU machine."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="requires Ubuntu/Linux, NCCL and at least two CUDA GPUs",
)
def test_two_gpu_training_resume_sampling_and_fid(image_dir, tmp_path):
    root = Path(__file__).resolve().parents[1]

    def run(output, steps, resume=None, standalone=False):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", DDPM_TEST_STEPS=str(steps), DDPM_TEST_MODEL="unet",
                   DDPM_TEST_FID="1", DDPM_TEST_DEVICE="cuda")
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        if resume:
            env["DDPM_TEST_RESUME"] = str(resume)
        worker = [str(root / "tests" / "ddp_worker.py"), str(image_dir), str(output)]
        for mode in (("train", "sample", "validate") if standalone else ("train",)):
            env["DDPM_TEST_MODE"] = mode
            result = subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", *worker],
                cwd=root, env=env, text=True, capture_output=True, timeout=300,
            )
            assert result.returncode == 0, result.stdout + "\n" + result.stderr
        assert (output / "rank-0.ok").exists() and (output / "rank-1.ok").exists()
        return torch.load(output / "weights" / "last.pkl", weights_only=False, map_location="cpu")

    continuous = run(tmp_path / "continuous", 2, standalone=True)
    split_dir = tmp_path / "split"
    saved = run(split_dir, 1)
    assert saved["batches_in_epoch"] == 1
    resumed = run(split_dir, 2, split_dir / "weights" / "last.pkl")
    assert abs(resumed["best_fid_score"] - continuous["best_fid_score"]) < 1e-3
    for name, value in continuous["model"].items():
        torch.testing.assert_close(resumed["model"][name], value, rtol=2e-5, atol=2e-5, msg=name)
    for name, value in continuous["ema"].items():
        torch.testing.assert_close(resumed["ema"][name], value, rtol=2e-5, atol=2e-5, msg=name)
    assert len(list((tmp_path / "continuous" / "inference").glob("*.png"))) == 5
    assert len(list((tmp_path / "continuous" / "validation").glob("*.png"))) == 4
