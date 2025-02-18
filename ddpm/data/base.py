# Copyright (c) Lornatang. All rights reserved.
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

__all__ = [
    "BaseImageFolder",
]


class BaseImageFolder(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions
        self.loader = loader

        self.samples = [path for extension in self.extensions for path in self.root.rglob(f"*{extension}")]

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            Any: where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
