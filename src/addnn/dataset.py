import functools
import multiprocessing
import os
import torch
import torchvision
import torchvision.datasets
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Callable, Iterable, List, Optional, Tuple


class Dataset:
    def __init__(self,
            train_set_constructor: Callable[..., torch.utils.data.Dataset],
            test_set_constructor: Callable[..., torch.utils.data.Dataset],
            input_size: Iterable[int],
            num_classes: int,
            train_set_subfolder: str = "",
            test_set_subfolder: str = "",
            train_set_transforms: Iterable[torch.nn.Module] = None,
            test_set_transforms: Iterable[torch.nn.Module] = None,
            mean: Optional[Iterable[float]] = None,
            std: Optional[Iterable[float]] = None,
            downloadable: bool = False):
        self._train_set_constructor = train_set_constructor
        self._test_set_constructor = test_set_constructor
        self._input_size = input_size
        self._num_classes = num_classes
        self._train_set_subfolder = train_set_subfolder
        self._test_set_subfolder = test_set_subfolder
        self._train_set_transforms = train_set_transforms
        self._test_set_transforms = test_set_transforms
        self._mean = mean
        self._std = std
        self._downloadable = downloadable

        test_set_normalization_transforms: List[torch.nn.Module] = []
        if self._test_set_transforms is not None:
            test_set_normalization_transforms.extend(self._test_set_transforms)
        test_set_normalization_transforms.append(transforms.ToTensor())
        test_set_normalization_transforms.append(transforms.Normalize(mean=self._mean, std=self._std))
        self._test_set_normalization = transforms.Compose(test_set_normalization_transforms)

    def load(self, path: str, batch_size: int, download: bool = False, num_workers: Optional[int] = None, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        # normalize the dataset based on the obtained mean and standard deviation
        train_set_path = os.path.join(path, self._train_set_subfolder)
        test_set_path = os.path.join(path, self._test_set_subfolder)

        train_set_transforms: List[torch.nn.Module] = []
        if self._train_set_transforms is not None:
            train_set_transforms.extend(self._train_set_transforms)
        train_set_transforms.append(transforms.ToTensor())
        train_set_transforms.append(transforms.Normalize(mean=self._mean, std=self._std))

        if self._downloadable:
            train_set_normalized = self._train_set_constructor(root=train_set_path,
                    transform=transforms.Compose(train_set_transforms), download=download)
        else:
            train_set_normalized = self._train_set_constructor(root=train_set_path,
                    transform=transforms.Compose(train_set_transforms))

        train_loader = DataLoader(train_set_normalized, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_set_normalized = self._test_set_constructor(root=test_set_path, transform=self._test_set_normalization)
        test_loader = DataLoader(test_set_normalized, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        return train_loader, test_loader

    @property
    def input_size(self) -> Iterable[int]:
        return self._input_size

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def test_set_normalization(self) -> torch.nn.Module:
        return self._test_set_normalization


datasets = dict()

datasets["imagenet"] = Dataset(
        torchvision.datasets.ImageFolder,
        torchvision.datasets.ImageFolder,
        input_size=(3, 224, 224),
        num_classes=1000,
        train_set_subfolder="train",
        test_set_subfolder="val",
        train_set_transforms=[transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()],
        test_set_transforms=[transforms.Resize(256), transforms.CenterCrop(224)],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        downloadable=False)

datasets["imagenet-hymenoptera"] = Dataset(
        torchvision.datasets.ImageFolder,
        torchvision.datasets.ImageFolder,
        input_size=(3, 224, 224),
        num_classes=2,
        train_set_subfolder="train",
        test_set_subfolder="val",
        train_set_transforms=[transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()],
        test_set_transforms=[transforms.Resize(256), transforms.CenterCrop(224)],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        downloadable=False)


datasets["mnist"] = Dataset(functools.partial(torchvision.datasets.MNIST, train=True),
        functools.partial(torchvision.datasets.MNIST, train=False),
        input_size=(1, 28, 28),
        num_classes=10,
        mean=[0.1307],
        std=[0.3081],
        downloadable=True)


datasets["fashion_mnist"] = Dataset(functools.partial(torchvision.datasets.FashionMNIST, train=True),
        functools.partial(torchvision.datasets.FashionMNIST, train=False),
        input_size=(1, 28, 28),
        num_classes=10,
        mean=[0.286],
        std=[0.353],
        downloadable=True)


datasets["cifar10"] = Dataset(functools.partial(torchvision.datasets.CIFAR10, train=True),
        functools.partial(torchvision.datasets.CIFAR10, train=False),
        input_size=(3, 32, 32),
        num_classes=10,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.247, 0.2435, 0.2616],
        downloadable=True)


datasets["cifar100"] = Dataset(functools.partial(torchvision.datasets.CIFAR100, train=True),
        functools.partial(torchvision.datasets.CIFAR100, train=False),
        input_size=(3, 32, 32),
        num_classes=100,
        mean=[0.5071, 0.4865, 0.4409],
        std=[0.2673, 0.2564, 0.2762],
        downloadable=True)

