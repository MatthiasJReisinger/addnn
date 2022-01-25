import addnn
import addnn.dataset
import addnn.model
import addnn.train
import click
import torch
from addnn.cli import cli
from addnn.util.entropy import normalized_entropy
from tqdm import tqdm
from typing import Any, List, cast

@cli.group()
def validate() -> None:
    pass

@validate.command("exits", help="Validate the DNN on a dataset and show the accuracy of its exit classifiers.")
@click.option("--dataset", "dataset_name", required=True, type=click.Choice(addnn.dataset.datasets.keys()), help="The dataset to use.")
@click.option("--dataset-root",
        default="./datasets",
        show_default=True,
        type=click.Path(exists=True),
        help="The path to the dataset to use.")
@click.option("--batch-size",
        type=int,
        required=False,
        default=1,
        show_default=True,
        help="The batch size to use to iterate the dataset.")
@click.argument("filename", type=click.Path(exists=True))
def exits(dataset_name: str, dataset_root: str, batch_size: int, filename: str) -> None:
    model = torch.load(filename)

    dataset = addnn.dataset.datasets[dataset_name]
    train_loader, test_loader = dataset.load(dataset_root, batch_size=batch_size, download=True)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    number_of_exits = sum([layer.exit_branch is not None for layer in model.layers])

    exits = []
    for layer in model.layers:
        if layer.exit_branch is not None:
            exits.append(layer.exit_branch)

    for exit_index in range(number_of_exits):
        criterion = torch.nn.CrossEntropyLoss()
        addnn.train.apply_test_set(model, test_loader, criterion, device, exit_index)

@validate.command("overall", help="Validate the DNN on a dataset and show its overall accuracy.")
@click.option("--dataset", "dataset_name", required=True, type=click.Choice(addnn.dataset.datasets.keys()), help="The dataset to use.")
@click.option("--dataset-root",
        default="./datasets",
        show_default=True,
        type=click.Path(exists=True),
        help="The path to the dataset to use.")
@click.argument("filename", type=click.Path(exists=True))
def overall(dataset_name: str, dataset_root: str, filename: str) -> None:
    model = torch.load(filename)

    dataset = addnn.dataset.datasets[dataset_name]
    train_loader, test_loader = dataset.load(dataset_root, batch_size=1, download=True)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    number_of_exits = sum([layer.exit_branch is not None for layer in model.layers])

    exits = []
    for layer in model.layers:
        if layer.exit_branch is not None:
            exits.append(layer.exit_branch)

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_corrects = torch.tensor(0)

    for inputs, labels in tqdm(test_loader, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            all_exit_outputs = model(inputs)

            for exit_index, exit_outputs in enumerate(all_exit_outputs):
                took_exit = False
                for exit_output in exit_outputs:
                    exit_output = exit_output.unsqueeze(0)
                    exit_entropy = normalized_entropy(exit_output[0])
                    if exit_entropy < exits[exit_index].confidence_threshold:
                        _, predictions = torch.max(exit_output, 1)
                        loss = criterion(exit_output, labels)
                        total_loss += loss.item() * inputs.size(0)
                        total_corrects += torch.sum(predictions == labels.data)
                        took_exit = True
                        break

                if took_exit:
                    break

    epoch_loss = total_loss / len(cast(List[Any], test_loader.dataset))
    epoch_accuracy = total_corrects.double() / len(cast(List[Any], test_loader.dataset))

    print('Overal test Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch_loss, epoch_accuracy))
