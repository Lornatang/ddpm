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
from .checkpoint import convert_optimizer_state_dict_to_half, load_checkpoint
from .ema import ModelEMA
from .envs import RANK, LOCAL_RANK, NUM_THREADS, set_seed_everything, unset_deterministic
from .events import LOGGER
from .ops import extract, num_to_groups, save_yaml
from .scheduler import cosine_beta_schedule, linear_beta_schedule, sigmoid_beta_schedule
from .torch_utils import de_parallel, get_gpu_info, is_parallel, select_device
