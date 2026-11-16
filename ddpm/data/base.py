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
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.transforms import functional as transform_functional

__all__ = [
    "BaseImageFolder", "ReproducibleImageFolder", "image_transform",
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

        if not self.root.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.root}")
        extensions = {extension.lower() for extension in self.extensions}
        self.samples = sorted(path for path in self.root.rglob("*")
                              if path.is_file() and path.suffix.lower() in extensions)

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


class ReproducibleImageFolder(BaseImageFolder):
    """Apply a deterministic flip for each (seed, epoch, image) tuple.

    The epoch is supplied as part of the sampler index, so this also works with
    persistent DataLoader workers and does not depend on worker RNG state.
    """

    def __init__(self, *args, seed: int = 0, augment: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.augment = augment

    def __getitem__(self, key):
        index, epoch = key
        sample = super().__getitem__(index)
        if self.augment:
            # SplitMix64 gives a stable, well-distributed decision without
            # consuming the model's or the workers' random number generators.
            mask = (1 << 64) - 1
            value = (self.seed + index + epoch * 0x9E3779B97F4A7C15) & mask
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
            value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
            value ^= value >> 31
            if value & 1:
                sample = transform_functional.hflip(sample)
        return sample


def image_transform(image_size, *, augment: bool = False, normalize: bool = False):
    """Build matching train/evaluation image preprocessing pipelines."""
    steps = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    if augment:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    if normalize:
        steps.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
    return transforms.Compose(steps)
