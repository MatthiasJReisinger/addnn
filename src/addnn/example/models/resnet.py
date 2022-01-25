import addnn
import addnn.dataset
import addnn.model
import addnn.train
import torch
import torchvision
from addnn.example.models.model_factory import ModelFactory
from typing import Callable, Optional


class SingleExitModelFactory(ModelFactory):
    def __init__(self, resnet_factory_function: Callable[..., torchvision.models.resnet.ResNet]):
        super().__init__()
        self._resnet_factory_function = resnet_factory_function

    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        # ignore epochs since we load the pretrained model that is already included in torchvision
        resnet = self._resnet_factory_function(pretrained=pretrained)
        return _transform_to_addnn_model(resnet)


class TwoExitModelFactory(ModelFactory):
    def __init__(self, resnet_factory_function: Callable[..., torchvision.models.resnet.ResNet], expansion: int):
        super().__init__()
        self._resnet_factory_function = resnet_factory_function
        self._expansion = expansion

    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        resnet = self._resnet_factory_function(pretrained=pretrained)

        # replace input layer to make it compatible with fashion MNIST
        number_of_color_channels = list(dataset.input_size)[0]
        if number_of_color_channels != 3:
            resnet.conv1 = torch.nn.Conv2d(number_of_color_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        model = _transform_to_multi_exit_addnn_model(resnet,
                                                     pretrained,
                                                     self._expansion,
                                                     dataset,
                                                     dataset_root,
                                                     download,
                                                     epochs,
                                                     batch_size,
                                                     num_workers,
                                                     num_exits=2)
        return model


class MultiExitModelFactory(ModelFactory):
    def __init__(self, resnet_factory_function: Callable[..., torchvision.models.resnet.ResNet], expansion: int):
        super().__init__()
        self._resnet_factory_function = resnet_factory_function
        self._expansion = expansion

    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        resnet = self._resnet_factory_function(pretrained=pretrained)

        # replace input layer to make it compatible with fashion MNIST
        number_of_color_channels = list(dataset.input_size)[0]
        if number_of_color_channels != 3:
            resnet.conv1 = torch.nn.Conv2d(number_of_color_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        model = _transform_to_multi_exit_addnn_model(resnet,
                                                     pretrained,
                                                     self._expansion,
                                                     dataset,
                                                     dataset_root,
                                                     download,
                                                     epochs,
                                                     batch_size,
                                                     num_workers,
                                                     num_exits=4)
        return model


def _transform_to_addnn_model(resnet: torchvision.models.resnet.ResNet) -> addnn.model.Model:
    layers = []
    layers.append(addnn.model.Layer(resnet.conv1))
    layers.append(addnn.model.Layer(resnet.bn1))
    layers.append(addnn.model.Layer(resnet.relu))
    layers.append(addnn.model.Layer(resnet.maxpool))

    layers.extend([addnn.model.Layer(layer) for layer in resnet.layer1])
    layers.extend([addnn.model.Layer(layer) for layer in resnet.layer2])
    layers.extend([addnn.model.Layer(layer) for layer in resnet.layer3])
    layers.extend([addnn.model.Layer(layer) for layer in resnet.layer4])

    classifier = torch.nn.Sequential(resnet.avgpool, torch.nn.Flatten(), resnet.fc)
    output_layer = addnn.model.Layer(None, addnn.model.Exit(classifier, 1.0))
    layers.append(output_layer)

    addnn_model = addnn.model.Model(layers)
    return addnn_model


def _transform_to_multi_exit_addnn_model(resnet: torchvision.models.resnet.ResNet, pretrained: bool, expansion: int,
                                         dataset: addnn.dataset.Dataset, dataset_root: str, download: bool, epochs: int,
                                         batch_size: int, num_workers: Optional[int],
                                         num_exits: int) -> addnn.model.Model:
    # only train the newly added classifiers, not the backbone network
    for parameter in resnet.parameters():
        parameter.requires_grad = False

    layers = []
    layers.append(addnn.model.Layer(resnet.conv1))
    layers.append(addnn.model.Layer(resnet.bn1))
    layers.append(addnn.model.Layer(resnet.relu))
    layers.append(addnn.model.Layer(resnet.maxpool))

    layers.extend([addnn.model.Layer(layer) for layer in resnet.layer1])

    if num_exits > 2:
        classifier = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
                                         torch.nn.Linear(64 * expansion, dataset.num_classes))
        layers.append(addnn.model.Layer(None, addnn.model.Exit(classifier, 0.8)))

    layers.extend([addnn.model.Layer(layer) for layer in resnet.layer2])

    if num_exits > 1:
        classifier = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
                                         torch.nn.Linear(128 * expansion, dataset.num_classes))
        layers.append(addnn.model.Layer(None, addnn.model.Exit(classifier, 0.8)))

    layers.extend([addnn.model.Layer(layer) for layer in resnet.layer3])

    if num_exits > 3:
        classifier = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
                                         torch.nn.Linear(256 * expansion, dataset.num_classes))
        layers.append(addnn.model.Layer(None, addnn.model.Exit(classifier, 0.8)))

    layers.extend([addnn.model.Layer(layer) for layer in resnet.layer4])

    classifier = torch.nn.Sequential(resnet.avgpool, torch.nn.Flatten(),
                                     torch.nn.Linear(512 * expansion, dataset.num_classes))
    output_layer = addnn.model.Layer(None, addnn.model.Exit(classifier, 1.0))
    layers.append(output_layer)

    addnn_model = addnn.model.Model(layers)

    if pretrained:
        train_loader, test_loader = dataset.load(dataset_root, batch_size, download, num_workers)
        addnn.train.train(addnn_model, train_loader, test_loader, epochs=epochs)

    return addnn_model
