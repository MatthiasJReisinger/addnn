import addnn
import addnn.model
import torch
import torchvision
from addnn.example.models.model_factory import ModelFactory
from typing import Optional


class SmallModelFactory(ModelFactory):
    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        # ignore epochs since we load the pretrained model that is already included in torchvision
        mobilenet = torchvision.models.mobilenetv3.mobilenet_v3_small(pretrained)
        return _transform_to_addnn_model(mobilenet)


class LargeModelFactory(ModelFactory):
    def generate_model(self, pretrained: bool, dataset: addnn.dataset.Dataset, dataset_root: str, download: bool,
                       epochs: int, batch_size: int, num_workers: Optional[int]) -> addnn.model.Model:
        # ignore epochs since we load the pretrained model that is already included in torchvision
        mobilenet = torchvision.models.mobilenetv3.mobilenet_v3_large(pretrained)
        return _transform_to_addnn_model(mobilenet)


def _transform_to_addnn_model(mobilenet: torchvision.models.mobilenetv3.MobileNetV3) -> addnn.model.Model:
    layers = [addnn.model.Layer(layer) for layer in mobilenet.features]
    layers.append(addnn.model.Layer(mobilenet.avgpool))
    layers.append(addnn.model.Layer(torch.nn.Flatten()))
    layers.append(addnn.model.Layer(None, addnn.model.Exit(mobilenet.classifier, 1.0)))
    addnn_model = addnn.model.Model(layers)
    return addnn_model
