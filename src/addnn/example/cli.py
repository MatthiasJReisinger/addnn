import addnn
import click
import torch
import torchvision
from addnn.example.models import resnet, vgg, mobilenetv3, googlenet, inception_v3
from addnn.cli import cli
from typing import Dict, Optional

model_factories: Dict[str, addnn.example.models.model_factory.ModelFactory] = dict()
model_factories["resnet18"] = resnet.SingleExitModelFactory(torchvision.models.resnet.resnet18)
model_factories["resnet34"] = resnet.SingleExitModelFactory(torchvision.models.resnet.resnet34)
model_factories["resnet50"] = resnet.SingleExitModelFactory(torchvision.models.resnet.resnet50)
model_factories["resnet101"] = resnet.SingleExitModelFactory(torchvision.models.resnet.resnet101)
model_factories["resnet152"] = resnet.SingleExitModelFactory(torchvision.models.resnet.resnet152)
model_factories["resnet152_2exits"] = resnet.TwoExitModelFactory(torchvision.models.resnet.resnet152, expansion=4)
model_factories["resnet18_multiexit"] = resnet.MultiExitModelFactory(torchvision.models.resnet.resnet18, expansion=1)
model_factories["resnet34_multiexit"] = resnet.MultiExitModelFactory(torchvision.models.resnet.resnet34, expansion=1)
model_factories["resnet50_multiexit"] = resnet.MultiExitModelFactory(torchvision.models.resnet.resnet50, expansion=4)
model_factories["resnet101_multiexit"] = resnet.MultiExitModelFactory(torchvision.models.resnet.resnet101, expansion=4)
model_factories["resnet152_multiexit"] = resnet.MultiExitModelFactory(torchvision.models.resnet.resnet152, expansion=4)
model_factories["vgg11"] = vgg.SingleExitModelFactory(torchvision.models.vgg.vgg11)
model_factories["vgg11_bn"] = vgg.SingleExitModelFactory(torchvision.models.vgg.vgg11_bn)
model_factories["vgg13"] = vgg.SingleExitModelFactory(torchvision.models.vgg.vgg13)
model_factories["vgg13_bn"] = vgg.SingleExitModelFactory(torchvision.models.vgg.vgg13_bn)
model_factories["vgg13_2exits"] = vgg.MultiExitModelFactory(torchvision.models.vgg.vgg13, side_exit_layer_index=20)
model_factories["vgg16"] = vgg.SingleExitModelFactory(torchvision.models.vgg.vgg16)
model_factories["vgg16_bn"] = vgg.SingleExitModelFactory(torchvision.models.vgg.vgg16_bn)
model_factories["vgg16_2exits"] = vgg.MultiExitModelFactory(torchvision.models.vgg.vgg16, side_exit_layer_index=24)
model_factories["vgg19"] = vgg.SingleExitModelFactory(torchvision.models.vgg.vgg19)
model_factories["vgg19_bn"] = vgg.SingleExitModelFactory(torchvision.models.vgg.vgg19_bn)
model_factories["vgg19_2exits"] = vgg.MultiExitModelFactory(torchvision.models.vgg.vgg19, side_exit_layer_index=28)
model_factories["mobilenet_v3_small"] = mobilenetv3.SmallModelFactory()
model_factories["mobilenet_v3_large"] = mobilenetv3.LargeModelFactory()
model_factories["googlenet"] = googlenet.SingleExitModelFactory()
model_factories["googlenet_multiexit"] = googlenet.MultiExitModelFactory()
model_factories["inception"] = inception_v3.SingleExitModelFactory()
model_factories["inception_multiexit"] = inception_v3.MultiExitModelFactory()


@cli.command("example", help="Export one of the predefined ADDNN model examples to the specified path.")
@click.option("--model", required=True, type=click.Choice(model_factories.keys()), help="The model to export.")
@click.option("--dataset",
              "dataset_name",
              required=True,
              type=click.Choice(addnn.dataset.datasets.keys()),
              help="The dataset the model should be compatible with and which should be used for training.")
@click.option("--dataset-root",
              default="./datasets",
              show_default=True,
              type=click.Path(exists=True),
              help="The path to the dataset to use.")
@click.option("--download",
              is_flag=True,
              default=False,
              help="Download the dataset if it's not available at the dataset root path.")
@click.option("--pretrained", is_flag=True, help="Export a pretrained version of the model.")
@click.option("--epochs",
              type=int,
              default=50,
              show_default=True,
              help="The number of training epochs to use when --pretrained is set.")
@click.option("--num-workers",
              type=int,
              required=False,
              help="The number of training workers to use when --pretrained is set.")
@click.option("--batch-size",
              type=int,
              default=1,
              show_default=True,
              help="The batch size to use for loading the dataset when --pretrained is set.")
@click.argument("path", type=click.Path())
def run(model: str, dataset_name: str, dataset_root: str, download: bool, pretrained: bool, epochs: int,
        num_workers: Optional[int], batch_size: int, path: str) -> None:
    model_generator = model_factories[model]
    dataset = addnn.dataset.datasets[dataset_name]
    addnn_model = model_generator.generate_model(pretrained, dataset, dataset_root, download, epochs, batch_size,
                                                 num_workers)

    # pass a random input through the model so that each layer can initialize its input size
    input_dimensions = dataset.input_size
    x = torch.rand(*input_dimensions)
    x = x.unsqueeze(0)
    addnn_model.cpu()
    addnn_model.eval()
    addnn_model(x)

    print("Input size of exported model: {}".format(input_dimensions))

    torch.save(addnn_model, path)
