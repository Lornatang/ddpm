# Copyright (c) Lornatang. All rights reserved.
import os
from typing import Union

import torch
from torch import nn

from .envs import NUM_THREADS
from .events import LOGGER

__all__ = [
    "de_parallel", "get_gpu_info", "is_parallel", "select_device",
]


def de_parallel(model: nn.Module) -> nn.Module:
    """Remove the distributed state from a model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        nn.Module: The model without the distributed state.
    """
    return model.module if is_parallel(model) else model


def get_gpu_info(index: int) -> str:
    """Get information about the GPU at the specified index.

    Args:
        index (int): The index of the GPU.

    Returns:
        str: A string describing the name and total memory size (in MiB) of the specified GPU.
    """
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def is_parallel(model: nn.Module) -> bool:
    """Check if the model is in a distributed state.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        bool: True if the model is distributed, False otherwise.
    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def select_device(device: Union[str, torch.device] = "", batch_size: int = 0, verbose: bool = True) -> torch.device:
    """Select the device index.

    Args:
        device (Union[str, torch.device], optional): The device index, can be "cpu" or "0", "0,1", etc. Defaults to "".
        batch_size (int, optional): The batch size. Defaults to 0.
        verbose (bool, optional): Whether to print device information. Defaults to True.

    Returns:
        torch.device: The selected device index.
    """
    # If the input is a torch.device object, return it directly
    if isinstance(device, torch.device):
        return device

    # Remove spaces, parentheses, etc.
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")

    # Determine the device type
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:  # Otherwise, use CUDA devices
        if device == "cuda":
            device = "0"
        # If there are multiple devices, remove empty devices
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])
        # Get the currently visible CUDA devices
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        # Set CUDA_VISIBLE_DEVICES to the selected devices
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        # Check if CUDA is available and if the number of devices is sufficient
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            raise ValueError(f"Incompatible device: {device}, CUDA_VISIBLE_DEVICES: {visible}")

    # Initialize the message string
    message = ""
    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(",") if device else "0"
        num_device = len(devices)
        # If there are multiple GPUs and the batch size is not a multiple of the number of GPUs, raise an error
        if num_device > 1:
            if batch_size < 1:
                raise ValueError("Please specify a valid batch size, i.e. batch=16.")
            if batch_size >= 0 and batch_size % num_device != 0:
                raise ValueError(
                    f"'batch_size={batch_size}' must be a multiple of GPU count {num_device}. Try 'batch_size={batch_size // num_device * num_device}' or "
                    f"'batch_size={batch_size // num_device * num_device + num_device}', the nearest batch sizes evenly divisible by {num_device}."
                )
        space = " " * (len(message) + 1)
        # Append detailed information of each GPU to the message
        for index, device_index in enumerate(devices):
            message += f"{'' if index == 0 else space}CUDA:{device_index} ({get_gpu_info(index)})\n"
        device_type = "cuda:0"
    elif mps and torch.backends.mps.is_available():
        message += f"Using MPS\n"
        device_type = "mps"
    else:
        message += f"Using CPU\n"
        device_type = "cpu"

    if device_type in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)

    if verbose:
        LOGGER.info(message.rstrip())

    return torch.device(device_type)
