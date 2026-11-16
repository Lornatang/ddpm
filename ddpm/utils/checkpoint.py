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
"""Checkpoint serialization and reconstruction for the built-in model."""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

__all__ = ["convert_optimizer_state_dict_to_half", "load_checkpoint", "model_config"]


def convert_optimizer_state_dict_to_half(state_dict: dict) -> dict:
    """Convert optimizer tensors in place; not suitable for resumable training."""
    for state in state_dict["state"].values():
        for key, value in state.items():
            if key != "step" and isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                state[key] = value.half()
    return state_dict


def model_config(model: nn.Module) -> Optional[dict]:
    """Return a portable constructor description for the built-in diffusion model.

    Custom models can still be resumed by passing an instance to load_checkpoint.
    """
    from ddpm.models.gaussian_diffusion import GaussianDiffusion
    from ddpm.models.unet import UNet

    if isinstance(model, GaussianDiffusion) and isinstance(model.model, UNet):
        _upgrade_legacy_model(model)
        return {"type": "gaussian_diffusion_unet", "diffusion": model.config, "unet": model.model.config}
    return None


def _upgrade_legacy_model(model: nn.Module) -> None:
    """Populate constructor metadata missing from the original pickled models."""
    from ddpm.models.gaussian_diffusion import GaussianDiffusion
    from ddpm.models.unet import UNet

    if not isinstance(model, GaussianDiffusion) or not isinstance(model.model, UNet):
        return
    unet = model.model
    if not hasattr(unet, "config"):
        output_dims = []
        for stage in unet.down_sampling:
            down = stage[-1]
            projection = down.downsample[-1] if hasattr(down, "downsample") else down
            output_dims.append(projection.out_channels)
        if any(dim % unet.dim for dim in output_dims):
            raise ValueError("Cannot infer channel multipliers from legacy checkpoint")
        heads = unet.mid_attn.heads
        unet.config = {
            "dim": unet.dim, "init_dim": unet.init_conv.out_channels,
            "out_dim": unet.out_dim, "in_channels": unet.in_channels,
            "channels_multipliers": tuple(dim // unet.dim for dim in output_dims),
            "self_condition": unet.self_condition, "learned_variance": unet.out_dim == 2 * unet.in_channels,
            "sinusoidal_pos_emb_theta": unet.time_mlp[0].theta,
            "dropout": unet.down_sampling[0][0].block1.dropout.p,
            "attn_dim_head": unet.mid_attn.to_qkv.out_channels // (3 * heads),
            "attn_heads": heads, "flash": unet.mid_attn.attend.flash,
        }
    if not hasattr(model, "learned_variance"):
        model.learned_variance = unet.out_dim == 2 * unet.in_channels
    if not hasattr(model, "config"):
        # The actual loss-weight buffer is restored from the state dict, including
        # any old min-SNR setting whose constructor arguments were not saved.
        model.config = {
            "image_size": model.image_size, "time_steps": model.num_time_steps,
            "sampling_time_steps": model.sampling_time_steps, "ddim_sampling_eta": model.ddim_sampling_eta,
            "offset_noise_strength": model.offset_noise_strength, "min_snr_loss_weight": False,
            "min_snr_gamma": 5, "immiscible": model.immiscible,
        }


def load_checkpoint(
    weights_path: Union[str, Path],
    device: Union[str, torch.device, None] = None,
    model: Optional[nn.Module] = None,
    prefer_ema: bool = True,
) -> Tuple[Any, dict]:
    """Load a checkpoint, reconstructing built-in models from their configuration.

    Legacy checkpoints contain pickled modules; only load files from trusted sources.
    """
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema") if prefer_ema else checkpoint.get("model")
    if state is None:
        state = checkpoint.get("model") if prefer_ema else checkpoint.get("ema")
    if isinstance(state, nn.Module):
        _upgrade_legacy_model(state)
        if model is None:
            model = state.float()
        else:
            model.load_state_dict(state.float().state_dict())
    elif isinstance(state, dict):
        if model is None:
            config = checkpoint.get("model_config")
            if not config or config.get("type") != "gaussian_diffusion_unet":
                raise ValueError("This checkpoint requires a model instance to reconstruct it")
            from ddpm.models.gaussian_diffusion import GaussianDiffusion
            from ddpm.models.unet import UNet

            model = GaussianDiffusion(UNet(**config["unet"]), **config["diffusion"])
        model.load_state_dict(state)
    else:
        raise ValueError("Checkpoint does not contain model weights")
    model = model.to(device).float().eval()
    return model, checkpoint
