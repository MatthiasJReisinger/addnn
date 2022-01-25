import addnn
import click
import torch
from addnn.cli import cli


@cli.command("torchhub", help="Export a raw torch model from pytorch.org/hub.")
@click.option("--repo", required=True, help="e.g. 'pytorch/vision:v0.9.0'.")
@click.option("--model", "model_name", required=True, help="e.g. 'resnet18'.")
@click.option("--pretrained", is_flag=True, help="Export a pretrained version of the model.")
@click.option("--torchscript", is_flag=True, help="Export the model as TorchScript.")
@click.argument("filename", type=click.Path(exists=False))
def run(repo: str, model_name: str, pretrained: bool, torchscript: bool, filename: str) -> None:
    model = torch.hub.load(repo, model_name, pretrained=pretrained)

    if torchscript:
        input_dimensions = (3, 224, 224)  # models from pytorch.org/hub are usually ImageNet-compatible
        x = torch.rand(*input_dimensions)
        x = x.unsqueeze(0)  # add a batch dimension
        model.eval()
        script = torch.jit.trace(model, x)  # transform module into TorchScript
        torch.jit.save(script, filename)
    else:
        torch.save(model, filename)

    print("Saved model to {}".format(filename))
