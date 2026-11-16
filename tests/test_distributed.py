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
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="PyTorch was built without distributed gloo support")
def test_torchrun_two_workers_shard_data_and_average_gradients(image_dir, tmp_path):
    root = Path(__file__).resolve().parents[1]
    def run(output, steps, resume=None, model="toy"):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        env["DDPM_TEST_STEPS"] = str(steps)
        env["DDPM_TEST_MODEL"] = model
        if resume:
            env["DDPM_TEST_RESUME"] = str(resume)
        worker = [str(root / "tests" / "ddp_worker.py"), str(image_dir), str(output)]
        if os.name == "nt":
            # Some Windows CPU wheels lack the libuv backend used by torchrun.
            # Spawn the same rank environment manually; Ubuntu uses torchrun below.
            env["USE_LIBUV"] = "0"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = str(sock.getsockname()[1])
            processes = []
            for rank in range(2):
                rank_env = dict(env, MASTER_ADDR="127.0.0.1", MASTER_PORT=port,
                                WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
                processes.append(subprocess.Popen([sys.executable, *worker], cwd=root, env=rank_env,
                                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            try:
                for process in processes:
                    stdout, stderr = process.communicate(timeout=120)
                    assert process.returncode == 0, stdout + "\n" + stderr
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
        else:
            result = subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", *worker],
                cwd=root, env=env, text=True, capture_output=True, timeout=120,
            )
            assert result.returncode == 0, result.stdout + "\n" + result.stderr
        assert (output / "rank-0.ok").exists() and (output / "rank-1.ok").exists()
        return torch.load(output / "weights" / "last.pkl", weights_only=False, map_location="cpu")

    continuous = run(tmp_path / "continuous", 2)
    split_dir = tmp_path / "split"
    run(split_dir, 1)
    resumed = run(split_dir, 2, split_dir / "weights" / "last.pkl")
    assert resumed["step"] == 2 and len(resumed["rng_states"]) == 2
    torch.testing.assert_close(resumed["model"]["weight"], continuous["model"]["weight"], rtol=0, atol=0)
    torch.testing.assert_close(resumed["ema"]["weight"], continuous["ema"]["weight"], rtol=0, atol=0)
    unet_checkpoint = run(tmp_path / "unet", 1, model="unet")
    assert unet_checkpoint["model_config"]["type"] == "gaussian_diffusion_unet"
