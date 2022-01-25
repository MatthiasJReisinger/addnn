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
        return _create_inception(pretrained=pretrained)


class MultiExitModelFactory(ModelFactory):
    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        model = _create_inception(pretrained=False, num_classes=dataset.num_classes)

        number_of_color_channels = list(dataset.input_size)[0]
        if number_of_color_channels != 3:
            raise Exception("Inceptionv3 only supports images with 3 color channels")

        if pretrained:
            train_loader, test_loader = dataset.load(dataset_root, batch_size=32)
            addnn.train.train(model, train_loader, test_loader, epochs=epochs)

        return model


def _create_inception(pretrained: bool, num_classes: int = 1000) -> addnn.model.Model:
    inception = torchvision.models.inception_v3(pretrained=pretrained,
                                                aux_logits=True,
                                                init_weights=False,
                                                num_classes=num_classes,
                                                transform_input=True)

    layers = []
    layers.append(addnn.model.Layer(inception.Conv2d_1a_3x3))
    layers.append(addnn.model.Layer(inception.Conv2d_2a_3x3))
    layers.append(addnn.model.Layer(inception.Conv2d_2b_3x3))
    layers.append(addnn.model.Layer(inception.maxpool1))
    layers.append(addnn.model.Layer(inception.Conv2d_3b_1x1))
    layers.append(addnn.model.Layer(inception.Conv2d_4a_3x3))
    layers.append(addnn.model.Layer(inception.maxpool2))
    layers.append(addnn.model.Layer(inception.Mixed_5b))
    layers.append(addnn.model.Layer(inception.Mixed_5c))
    layers.append(addnn.model.Layer(inception.Mixed_5d))
    layers.append(addnn.model.Layer(inception.Mixed_6a))
    layers.append(addnn.model.Layer(inception.Mixed_6b))
    layers.append(addnn.model.Layer(inception.Mixed_6c))
    layers.append(addnn.model.Layer(inception.Mixed_6d))
    layers.append(addnn.model.Layer(inception.Mixed_6e))
    layers.append(addnn.model.Layer(None, addnn.model.Exit(inception.AuxLogits, 0.8)))
    layers.append(addnn.model.Layer(inception.Mixed_7a))
    layers.append(addnn.model.Layer(inception.Mixed_7b))
    layers.append(addnn.model.Layer(inception.Mixed_7c))
    layers.append(addnn.model.Layer(inception.avgpool))
    layers.append(addnn.model.Layer(inception.dropout))
    layers.append(addnn.model.Layer(torch.nn.Flatten()))
    layers.append(addnn.model.Layer(None, addnn.model.Exit(inception.fc, 0.8)))

    addnn_model = addnn.model.Model(layers)

    return addnn_model
