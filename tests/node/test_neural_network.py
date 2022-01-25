import pytest
import torch
from addnn.model import Layer, Exit
from addnn.node.neural_network import Model, NeuralNetwork


class MockClassifier(torch.nn.Module):
    def __init__(self, exit_value: torch.Tensor):
        super().__init__()
        self._exit_value = exit_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._exit_value


def _create_mock_exit_layer(exit_value: torch.Tensor, confidence_threshold: float) -> Layer:
    exit_classifier = MockClassifier(exit_value)
    exit_branch = Exit(classifier=exit_classifier, confidence_threshold=confidence_threshold)
    layer = Layer(main_branch=None, exit_branch=exit_branch)
    return layer


def test_neural_network_takes_final_exit_if_no_side_exits_are_available() -> None:
    # given
    final_exit_value = torch.tensor([[0.0, 1.0]])
    final_exit_layer = _create_mock_exit_layer(final_exit_value, 1.0)
    layers = [final_exit_layer]
    model = Model(layers, remote_layer=None, start_layer_index=0)

    # when
    neural_network = NeuralNetwork()
    neural_network.model = model
    classification = neural_network.infer(torch.tensor(42.0))

    # then
    expected_class_label = final_exit_value.argmax(1)
    assert classification == expected_class_label


def test_neural_network_takes_final_exit_if_no_side_exits_are_deactivated() -> None:
    # given
    side_exit_value = torch.tensor([[1.0, 0.0]])
    side_exit_layer = _create_mock_exit_layer(side_exit_value, 0.0)
    final_exit_value = torch.tensor([[0.0, 1.0]])
    final_exit_layer = _create_mock_exit_layer(final_exit_value, 1.0)
    layers = [side_exit_layer, final_exit_layer]
    model = Model(layers, remote_layer=None, start_layer_index=0)

    # when
    neural_network = NeuralNetwork()
    neural_network.model = model
    classification = neural_network.infer(torch.tensor(42.0))

    # then
    expected_class_label = final_exit_value.argmax(1)
    assert classification == expected_class_label


def test_neural_network_takes_side_exit_if_entropy_is_below_threshold() -> None:
    # given
    side_exit_value = torch.tensor([[1.0, 0.0]])
    side_exit_layer = _create_mock_exit_layer(side_exit_value, 1.0)
    final_exit_value = torch.tensor([[0.0, 1.0]])
    final_exit_layer = _create_mock_exit_layer(final_exit_value, 1.0)
    layers = [side_exit_layer, final_exit_layer]
    model = Model(layers, remote_layer=None, start_layer_index=0)

    # when
    neural_network = NeuralNetwork()
    neural_network.model = model
    classification = neural_network.infer(torch.tensor(42.0))

    # then
    expected_class_label = side_exit_value.argmax(1)
    assert classification == expected_class_label
