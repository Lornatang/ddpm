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
import os
import random

import numpy as np
import torch
import torch.backends.mps

__all__ = [
    "RANK", "LOCAL_RANK", "NUM_THREADS",
    "set_seed_everything", "unset_deterministic",
]

# Get the values of environment variables "RANK" and "LOCAL_RANK", default to -1 if not present
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
# Set and limit the number of threads to use, minimum 1, maximum 8, and no more than CPU cores minus 1
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
# Configure environment variable to set the maximum number of threads for NUMEXPR
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)
# Set PyTorch print options, adjusting output line width, precision, and profile
torch.set_printoptions(linewidth=320, precision=4, profile="default")
# Set NumPy print options, adjusting output line width and float format
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})


def set_seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for CUDA, Numpy, and Python to ensure experiment reproducibility.

    Args:
        seed (int): Random seed
        deterministic (bool, optional): Whether to set CuDNN backend to deterministic. Defaults to False
    """
    # Set Python random seed
    random.seed(seed)

    # Set Numpy random seed
    np.random.seed(seed)

    # Set Torch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # To ensure experiment reproducibility, set CuDNN backend to deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        unset_deterministic()


def unset_deterministic() -> None:
    """Unsets all the configurations applied for deterministic training."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.pop("PYTHONHASHSEED", None)
