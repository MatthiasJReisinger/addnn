import addnn
import addnn.node
import addnn.node.neural_network
import click
import csv
import pickle
import random
import torch
from addnn.cli import cli
from addnn.profile.layer_profile import LayerProfile, get_layer_profiles
from addnn.util.torchscript import read_model_from_preprocessed_script
from timeit import timeit
from torch.utils.data import DataLoader
from typing import Iterable, Optional, Tuple


@cli.group()
def profile() -> None:
    pass


@profile.command("export", help="Export a profile for the given model.")
@click.option("--torchscript", is_flag=True, default=False, help="Whether the model is a TorchScript trace.")
@click.option("--input-shape", help="shape of the given model (e.g., '3,224,224').")
@click.option("--format",
              "export_format",
              required=True,
              default="pickle",
              type=click.Choice(["pickle", "csv"]),
              help="The format of the exported profile.")
@click.option("--out",
              type=click.Path(),
              required=True,
              default="model.profile",
              help="Where to save the profiling data.")
@click.argument("filename", type=click.Path(exists=True))
def export(torchscript: bool, input_shape: str, export_format: str, out: str, filename: str) -> None:
    if torchscript:
        model = read_model_from_preprocessed_script(filename, input_shape)
    else:
        model = torch.load(filename)
        model.eval()

    layers = model.layers
    layer_profiles = get_layer_profiles(layers)

    if export_format == "pickle":
        with open(out, "wb") as out_file:
            pickle.dump(layer_profiles, out_file)
    else:
        with open(out, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            title_row = ["layer", "memory", "storage", "input-size", "mflops", "execution-probability"]
            csvwriter.writerow(title_row)

            for layer_index in range(len(layer_profiles)):
                memory = "{:.2f}".format(float(layer_profiles[layer_index].in_memory_size) / 1000 / 1000)
                storage = "{:.2f}".format(float(layer_profiles[layer_index].storage_size) / 1000 / 1000)
                input_size = "{:.2f}".format(float(layer_profiles[layer_index].marshalled_input_size) / 1000 / 1000)
                mflops = "{:.2f}".format(float(layer_profiles[layer_index].flops) / 1000 / 1000)
                operators = " ".join(layer_profiles[layer_index].operators)
                row = [
                    layer_index, memory, storage, input_size, mflops, layer_profiles[layer_index].execution_probability,
                    operators
                ]
                csvwriter.writerow(row)


@profile.command("show", help="Show profile information for the given model.")
@click.option("--torchscript", is_flag=True, default=False)
@click.option("--csv", "export_csv", is_flag=True, default=False)
@click.option("--input-shape", help="shape of the given model (e.g., '3,224,224').")
@click.option("--is-profile", is_flag=True, default=False)
@click.argument("filename", type=click.Path(exists=True))
def show(filename: str, torchscript: bool, export_csv: bool, input_shape: str, is_profile: bool) -> None:
    if is_profile:
        with open(filename, "rb") as profile_file:
            layer_profiles = pickle.load(profile_file)
    else:
        if torchscript:
            model = read_model_from_preprocessed_script(filename, input_shape)
        else:
            model = torch.load(filename)
        layers = model.layers
        layer_profiles = get_layer_profiles(layers)

    model_memory = 0.0
    model_storage = 0.0
    model_flops = 0.0

    for layer_index in range(len(layer_profiles)):
        print("profile of layer {}:".format(layer_index))
        print("\tmemory:      {} MB".format(float(layer_profiles[layer_index].in_memory_size) / 1000 / 1000))
        print("\tstorage:     {} MB".format(float(layer_profiles[layer_index].storage_size) / 1000 / 1000))
        print("\tinput size:  {} MB".format(float(layer_profiles[layer_index].marshalled_input_size) / 1000 / 1000))
        print("\tMFLOPs:      {}".format(float(layer_profiles[layer_index].flops) / 1000 / 1000))
        print("\tOperators:   {}".format(", ".join(layer_profiles[layer_index].operators)))
        print("\texecution probability: {}".format(layer_profiles[layer_index].execution_probability))
        print()

        model_memory += layer_profiles[layer_index].in_memory_size
        model_storage += layer_profiles[layer_index].storage_size
        model_flops += layer_profiles[layer_index].flops

    print("total requirements of model:")
    print("\tmemory:  {} MB".format(model_memory / 1000 / 1000))
    print("\tstorage: {} MB".format(model_storage / 1000 / 1000))
    print("\tGFLOPs:  {}".format(model_flops / 1000 / 1000 / 1000))


@profile.command("local-random-throughput",
                 help="Profile the theoritcal throughput of the given model for random input on the current host.")
@click.option("--torchscript", is_flag=True, default=False, help="Whether the model is a TorchScript trace.")
@click.option("--input-shape",
              help="shape of the given model (e.g., '3,224,224'), only required in case of TorchScript.")
@click.option("--iterations", required=True, type=int, default=100)
@click.argument("filename", type=click.Path(exists=True))
def local_random_throughput(torchscript: bool, input_shape: Optional[str], iterations: int, filename: str) -> None:
    if torchscript:
        if input_shape is None:
            raise Exception("please provide the expected input shape of the given TorchScript model")

        model = read_model_from_preprocessed_script(filename, input_shape)
        dimensions = [int(dimension) for dimension in input_shape.split(",")]
    else:
        model = torch.load(filename)
        model.eval()
        if not model.layers[0].input_size:
            raise Exception("cannot infer throughput for untrained model")

        dimensions = list(model.layers[0].input_size)

    layers = model.layers

    neural_network = addnn.node.neural_network.NeuralNetwork()
    neural_network.model = addnn.node.neural_network.Model(layers, remote_layer=None, start_layer_index=0)

    time = timeit(lambda: _infer_random(dimensions, neural_network), number=iterations)
    throughput = iterations / time
    print("classified {} samples in {:.2f} seconds".format(iterations, time))
    print("throughput: {:.2f} samples/sec".format(throughput))


def _infer_random(dimensions: Iterable[int], neural_network: addnn.node.neural_network.NeuralNetwork) -> None:
    input_tensor = torch.rand(*dimensions)
    input_tensor = input_tensor.unsqueeze(0)
    neural_network.infer(input_tensor)


@profile.command("local-dataset-throughput",
                 help="Profile the theoritcal throughput of the given model for the given dataset on the current host.")
@click.option("--torchscript", is_flag=True, default=False, help="Whether the model is a TorchScript trace.")
@click.option("--input-shape",
              help="shape of the given model (e.g., '3,224,224'), only required in case of TorchScript.")
@click.option("--iterations", required=True, type=int, default=100)
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
@click.option("--seed", type=int, help="The seed to use for the random number generator.")
@click.argument("filename", type=click.Path(exists=True))
def local_dataset_throughput(torchscript: bool, input_shape: Optional[str], iterations: int, dataset_name: str,
                             dataset_root: str, seed: int, filename: str) -> None:
    if torchscript:
        if input_shape is None:
            raise Exception("please provide the expected input shape of the given TorchScript model")

        model = read_model_from_preprocessed_script(filename, input_shape)
    else:
        model = torch.load(filename)
        model.eval()

    layers = model.layers

    neural_network = addnn.node.neural_network.NeuralNetwork()
    neural_network.model = addnn.node.neural_network.Model(layers, remote_layer=None, start_layer_index=0)

    random.seed(seed)
    dataset = addnn.dataset.datasets[dataset_name]
    train_loader, test_loader = dataset.load(dataset_root, batch_size=1, download=True)
    data_iterator = iter(test_loader)

    time = timeit(lambda: _infer_dataset(neural_network, data_iterator), number=iterations)
    throughput = iterations / time
    latency = time / iterations
    print("classified {} samples in {:.2f} seconds".format(iterations, time))
    print("throughput: {:.2f} samples/sec".format(throughput))
    print("latency: {:.2f} sec/sample".format(latency))


def _infer_dataset(neural_network: addnn.node.neural_network.NeuralNetwork,
                   data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> None:
    input_tensor, label = next(data_loader)  # type: ignore
    neural_network.infer(input_tensor)
