import addnn
import addnn.dataset
import addnn.model
import torch
import torchvision
import sys
from addnn.example.models.model_factory import ModelFactory
from typing import Optional


class SingleExitModelFactory(ModelFactory):
    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        # ignore epochs since we load the pretrained model that is already included in torchvision
        return _create_googlenet(pretrained=pretrained)


class MultiExitModelFactory(ModelFactory):
    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        model = _create_googlenet(pretrained=False, num_classes=dataset.num_classes)

        # replace input layer to make it compatible with fashion MNIST
        conv2d = sys.modules["torchvision.models.googlenet"].BasicConv2d  # type: ignore

        number_of_color_channels = list(dataset.input_size)[0]
        if number_of_color_channels != 3:
            model.layers[0] = addnn.model.Layer(conv2d(number_of_color_channels, 64, kernel_size=7, stride=2,
                                                       padding=3))

        if pretrained:
            train_loader, test_loader = dataset.load(dataset_root, batch_size=32)
            addnn.train.train(model, train_loader, test_loader, epochs=epochs)

        return model


def _create_googlenet(pretrained: bool, num_classes: int = 1000) -> addnn.model.Model:
    googlenet = torchvision.models.googlenet(pretrained=False,
                                             aux_logits=True,
                                             init_weights=False,
                                             num_classes=num_classes)

    layers = []
    layers.append(addnn.model.Layer(googlenet.conv1))
    layers.append(addnn.model.Layer(googlenet.maxpool1))
    layers.append(addnn.model.Layer(googlenet.conv2))
    layers.append(addnn.model.Layer(googlenet.conv3))
    layers.append(addnn.model.Layer(googlenet.maxpool2))
    layers.append(addnn.model.Layer(googlenet.inception3a))
    layers.append(addnn.model.Layer(googlenet.inception3b))
    layers.append(addnn.model.Layer(googlenet.maxpool3))
    layers.append(addnn.model.Layer(googlenet.inception4a))
    layers.append(addnn.model.Layer(None, addnn.model.Exit(googlenet.aux1, 0.8)))
    layers.append(addnn.model.Layer(googlenet.inception4b))
    layers.append(addnn.model.Layer(googlenet.inception4c))
    layers.append(addnn.model.Layer(googlenet.inception4d))
    layers.append(addnn.model.Layer(None, addnn.model.Exit(googlenet.aux2, 0.8)))
    layers.append(addnn.model.Layer(googlenet.inception4e))
    layers.append(addnn.model.Layer(googlenet.maxpool4))
    layers.append(addnn.model.Layer(googlenet.inception5a))
    layers.append(addnn.model.Layer(googlenet.inception5b))
    layers.append(addnn.model.Layer(googlenet.avgpool))
    layers.append(addnn.model.Layer(torch.nn.Flatten()))
    layers.append(addnn.model.Layer(googlenet.dropout))
    layers.append(addnn.model.Layer(None, addnn.model.Exit(googlenet.fc, 1.0)))
    addnn_model = addnn.model.Model(layers)

    return addnn_model
