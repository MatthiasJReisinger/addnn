import torch
import torch.nn
import os
from collections.abc import Sequence
from typing import Iterable, List, Optional, Tuple


class Exit(torch.nn.Module):
    """Represents an exit branch of an Adaptive Distributed Deep Neural Network."""
    def __init__(self, classifier: torch.nn.Module, confidence_threshold: float):
        """Initializes an Exit."""
        super().__init__()
        self._classifier = classifier
        self._confidence_threshold = confidence_threshold
        self._number_of_exited_samples = 0

    @property
    def classifier(self) -> torch.nn.Module:
        """Returns the classifier of this ADDNN exit."""
        return self._classifier

    @property
    def confidence_threshold(self) -> float:
        """Returns the confidence threshold of this ADDNN exit."""
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, confidence_threshold: float) -> None:
        """Set the confidence threshold of this ADDNN exit."""
        self._confidence_threshold = confidence_threshold

    @property
    def number_of_exited_samples(self) -> int:
        """Returns the number of samples that took this exit."""
        return self._number_of_exited_samples

    @number_of_exited_samples.setter
    def number_of_exited_samples(self, number_of_exited_samples: int) -> None:
        """Set the number of samples that took this exit."""
        self._number_of_exited_samples = number_of_exited_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._classifier(x)


class Layer(torch.nn.Module):
    """Represents a layer of a Adaptive Distributed Deep Neural Network."""
    def __init__(self, main_branch: Optional[torch.nn.Module], exit_branch: Optional[Exit] = None):
        """Initializes a Layer."""
        super().__init__()

        self._main_branch = main_branch
        self._exit_branch = exit_branch
        self._input_size: Optional[torch.Size] = None

    @property
    def main_branch(self) -> Optional[torch.nn.Module]:
        """Returns the main branch of this ADDNN layer."""
        return self._main_branch

    @property
    def exit_branch(self) -> Optional[Exit]:
        """Returns the exit branch of this ADDNN layer."""
        return self._exit_branch

    @property
    def input_size(self) -> Optional[torch.Size]:
        """Returns the input size of the layer's main branch."""
        return self._input_size

    @property
    def number_of_exited_samples(self) -> int:
        """Returns the number of samples that took the exit at this layer."""
        if self._exit_branch is not None:
            return self._exit_branch.number_of_exited_samples
        else:
            return 0

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns the output of the main branch and the prediction of the exit branch."""

        if self._input_size is None:
            self._input_size = x.shape[1:]

        if self._main_branch is not None:
            main_output = self._main_branch(x)
        else:
            main_output = x

        exit_output = None
        if self._exit_branch is not None:
            exit_output = self._exit_branch(main_output)

        return (main_output, exit_output)


class Model(torch.nn.Module):
    """Represents the model of a Adaptive Distributed Deep Neural Network.

    This class acts as the main interface to the ADDNN for both the training phase and for serving the trained model to
    production.
    """
    def __init__(self, layers: Iterable[Layer]):
        """Initializes a Model.

        Args:
            modules: The layers that make up this ADDNN.
        """
        super().__init__()
        self._layers = torch.nn.ModuleList(layers)

    @property
    def layers(self) -> List[Layer]:
        """Returns the layers of this ADDNN model."""
        return self._layers  # type: ignore

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns the predictions of all exit classifiers in this model."""
        predictions = []

        for layer in self._layers:
            (main_output, exit_output) = layer(x)
            x = main_output

            if exit_output is not None:
                predictions.append(exit_output)

        return predictions
