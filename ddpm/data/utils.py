# Copyright (c) Lornatang. All rights reserved.
from typing import Any

from torch.utils.data import DataLoader

__all__ = [
    "cycle_data_loader"
]


def cycle_data_loader(data_loader: DataLoader) -> Any:
    """Cycle through a DataLoader indefinitely.

    Args:
        data_loader (DataLoader): The DataLoader to cycle through.
    """
    while True:
        for data in data_loader:
            yield data
