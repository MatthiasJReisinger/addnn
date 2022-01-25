import addnn
import addnn.model
import torch
import torchvision
from addnn.example.models.model_factory import ModelFactory
from typing import Callable, Optional


class SingleExitModelFactory(ModelFactory):
    def __init__(self, vgg_factory_function: Callable[..., torchvision.models.vgg.VGG]):
        super().__init__()
        self._vgg_factory_function = vgg_factory_function

    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        # ignore epochs since we load the pretrained model that is already included in torchvision
        vgg = self._vgg_factory_function(pretrained=pretrained)
        layers = []
        layers.extend([addnn.model.Layer(layer) for layer in vgg.features])
        layers.append(addnn.model.Layer(vgg.avgpool))
        layers.append(addnn.model.Layer(torch.nn.Flatten()))
        layers.extend([addnn.model.Layer(layer) for layer in vgg.classifier[:-1]])
        layers.append(addnn.model.Layer(None, addnn.model.Exit(vgg.classifier[-1], 1.0)))
        addnn_model = addnn.model.Model(layers)
        return addnn_model


class MultiExitModelFactory(ModelFactory):
    def __init__(self, vgg_factory_function: Callable[..., torchvision.models.vgg.VGG], side_exit_layer_index: int):
        super().__init__()
        self._vgg_factory_function = vgg_factory_function
        self._side_exit_layer_index = side_exit_layer_index

    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:

        vgg = self._vgg_factory_function(pretrained=pretrained)

        # only train the newly added classifiers, not the backbone network
        for parameter in vgg.parameters():
            parameter.requires_grad = False

        layers = []
        layers.extend([addnn.model.Layer(layer) for layer in vgg.features[:self._side_exit_layer_index]])
        layers.append(addnn.model.Layer(None, addnn.model.Exit(_create_side_classifier(512, dataset.num_classes), 0.0)))
        layers.extend([addnn.model.Layer(layer) for layer in vgg.features[self._side_exit_layer_index:]])

        layers.append(addnn.model.Layer(vgg.avgpool))
        layers.append(addnn.model.Layer(torch.nn.Flatten()))
        layers.extend([addnn.model.Layer(layer) for layer in vgg.classifier[:-1]])
        layers.append(addnn.model.Layer(None, addnn.model.Exit(torch.nn.Linear(4096, dataset.num_classes), 1.0)))

        addnn_model = addnn.model.Model(layers)

        if pretrained:
            train_loader, test_loader = dataset.load(dataset_root, batch_size, download, num_workers)
            addnn.train.train(addnn_model, train_loader, test_loader, epochs=epochs)

        return addnn_model


def _create_side_classifier(dim: int, num_classes: int, dropout: float = 0.5) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((7, 7)),
        torch.nn.Flatten(1),
        torch.nn.Linear(dim * 7 * 7, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(p=dropout),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(p=dropout),
        torch.nn.Linear(4096, num_classes),
    )
