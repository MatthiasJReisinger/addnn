import addnn
import addnn.dataset
import addnn.model
import click
import torch
from addnn.cli import cli
from addnn.util.entropy import normalized_entropy
from tqdm import tqdm
from typing import List, Optional, Tuple


@cli.group()
def thresholds() -> None:
    pass


@thresholds.command("learn-exit-rates", help="Learn exit rates of side exit classifiers based on test set.")
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
@click.option("--dry-run",
        is_flag=True,
        help="Do not save the exit rate in the model.")
@click.argument("filename", type=click.Path(exists=True))
def learn_exit_rates(dataset_name: str, dataset_root: str, batch_size: int, dry_run: bool, filename: str) -> None:
    model = torch.load(filename)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    model.to(device)
    model.eval()

    # reset the the counter for each exit to 0
    for layer in model.layers:
        if layer.exit_branch is not None:
            layer.exit_branch.number_of_exited_samples = 0

    # track the number of samples that leave at each exit
    dataset = addnn.dataset.datasets[dataset_name]
    train_loader, test_loader = dataset.load(dataset_root, batch_size=batch_size, download=True, shuffle=False)
    number_of_samples = 0
    for batch, target in tqdm(test_loader, leave=False):
        batch = batch.to(device)
        _track_number_of_samples_at_each_exit(model, batch)
        number_of_samples += len(batch)

    # print out the exit rates
    print("exit rates:")
    exit_index = 0
    for layer in model.layers:
        if layer.exit_branch is not None:
            exit_rate = layer.number_of_exited_samples / float(number_of_samples)
            print("exit {}: {}% ({}/{})".format(exit_index, exit_rate * 100.0, layer.number_of_exited_samples, number_of_samples))
            exit_index += 1

    if not dry_run:
        torch.save(model, filename)


def _track_number_of_samples_at_each_exit(model: addnn.model.Model, batch: torch.Tensor) -> None:
    x = batch
    for layer in model.layers:
        main_branch_output, batch_predictions = layer(x)
        if layer.exit_branch is not None:
            batch_for_next_layer = []
            sample_index = 0
            for prediction in batch_predictions:
                if normalized_entropy(prediction) <= layer.exit_branch.confidence_threshold:
                    layer.exit_branch.number_of_exited_samples += 1
                else:
                    batch_for_next_layer.append(main_branch_output[sample_index].unsqueeze(0))
                sample_index += 1

            # if all samples of the current batch exited at the current classifier, then there's no sample left that
            # has to be classified at later exits
            if len(batch_for_next_layer) == 0:
                return

            # concatenate the remaining samples of the current batch so that they can be passed to the next layer
            x = torch.cat(batch_for_next_layer)
        else:
            x = main_branch_output


@thresholds.command("set", help="Learn exit rates of side exit classifiers based on test set.")
@click.option("--exit",
        "exit_thresholds",
        required=True,
        type=(int, float),
        multiple=True,
        help="An exit index and its new confidence threshold value.")
@click.argument("filename", type=click.Path(exists=True))
def set(exit_thresholds: List[Tuple[int, float]], filename: str) -> None:
    model = torch.load(filename)

    exit_threshold_map = dict()
    for exit_index, confidence_threshold in exit_thresholds:
        exit_threshold_map[exit_index] = confidence_threshold

    exit_index = 0
    for layer in model.layers:
        if layer.exit_branch is not None:
            if exit_index in exit_threshold_map:
                new_confidence_threshold = exit_threshold_map[exit_index]
                layer.exit_branch.confidence_threshold = new_confidence_threshold
            exit_index += 1

    torch.save(model, filename)


@thresholds.command("show", help="Show the confidence thresholds of the exit classifiers in the given model.")
@click.argument("filename", type=click.Path(exists=True))
def show(filename: str) -> None:
    model = torch.load(filename)

    exit_index = 0
    for layer in model.layers:
        if layer.exit_branch is not None:
            print("Exit {}: {}".format(exit_index, layer.exit_branch.confidence_threshold))
            exit_index += 1
