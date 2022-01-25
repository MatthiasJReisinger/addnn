import addnn
import addnn.model
from abc import ABC, abstractmethod
from typing import Iterable, Optional


class ModelFactory(ABC):
    """
    Utility base class for providing a common interface for creating different kinds of (pretrained) ADDNN models.
    """
    @abstractmethod
    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool, epochs: int,
                       batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        """Create a model instance.

        Args:
            pretrained: Whether to create a pretrained model instance.
            dataset: The dataset the model should be compatible with and which should be used for training.
            dataset_root: The path to the dataset.
            download: Whether to download the dataset if it's not available at the dataset root path.
            epochs: The number of training epochs to use when creating a pretrained model.
            batch_size: The batch size for loading the dataset when creating a pretrained model.
            num_workers: The number of workers to use for training.
        """
        raise NotImplementedError
