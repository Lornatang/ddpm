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
"""Process and device setup for torchrun sampling and validation."""

import os
from datetime import timedelta

import torch
from torch import distributed as dist


class DistributedInference:
    """Select a local device and manage the process group for sampling or evaluation."""

    def __init__(self, device):
        """Configure the requested device and any torchrun process group.

        Args:
            device (str | torch.device): CPU, MPS, or CUDA device specification. Multi-rank CUDA runs use
                ``LOCAL_RANK`` as the visible device index.

        Raises:
            ValueError: If the rank, world size, CUDA device, or existing process group is incompatible.
        """
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid torchrun RANK or WORLD_SIZE")
        self.is_main = self.rank == 0
        self.owns_group = False
        requested = str(device).lower()
        if requested == "cpu":
            self.device = torch.device("cpu")
        elif requested == "mps" and self.world_size == 1 and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            if self.world_size > 1:
                index = self.local_rank
            elif requested in ("", "auto", "cuda"):
                index = 0
            elif requested.startswith("cuda:"):
                index = int(requested.split(":", 1)[1])
            else:
                index = int(requested)
            if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
                raise ValueError(f"CUDA device {index} is unavailable")
            self.device = torch.device("cuda", index)
            torch.cuda.set_device(self.device)
        if self.world_size == 1:
            return
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if self.device.type == "cuda" else "gloo",
                init_method="env://", timeout=timedelta(hours=12),
            )
            self.owns_group = True
        if dist.get_world_size() != self.world_size or dist.get_rank() != self.rank:
            raise ValueError("Existing process group does not match the torchrun environment")
        # A shared seed must not make different ranks generate identical images.
        torch.manual_seed((torch.initial_seed() + self.rank) % (1 << 63))

    def close(self):
        """Destroy the process group if this instance created it."""
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
