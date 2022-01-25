import addnn
import addnn.serve.cli
import click
import csv
import gc
import grpc
import multiprocessing
import pickle
import pulp
import random
import os
import statistics
import torch
import time
from addnn.benchmark.cli import benchmark
from addnn.benchmark.strategies.executor import BenchmarkResult, Executor
from addnn.benchmark.strategies.scenarios import scenario_iot, scenario_iot_edge_cloud
from addnn.cli import cli
from addnn.controller.proto import controller_pb2
from addnn.example.models import resnet, mobilenetv3, vgg
from addnn.node.proto import node_pb2, node_state_pb2
from addnn.profile.layer_profile import LayerProfile, get_layer_profiles
from addnn.serve.placement import strategy_loader
from addnn.serve.placement.placement import Placement, estimate_ent_to_end_latency
from addnn.util.torchscript import read_model_from_preprocessed_script
from matplotlib import pyplot
from typing import Any, Dict, Iterable, List, Optional, Tuple

scenarios = dict()
scenarios["iot"] = scenario_iot
scenarios["iot_edge_cloud"] = scenario_iot_edge_cloud


@benchmark.group(help="Placement strategy benchmarks.")
def strategies() -> None:
    pass


@strategies.command("vary-hierarchy-size",
                    help="Benchmark placement algorithms on a fixed model and vary the size of the compute hierarchy.")
@click.option("--scenario", required=True, type=click.Choice(scenarios.keys()), help="The benchmark scenario.")
@click.option("--min-num-nodes", required=True, type=int, help="The minimal number of nodes in the hierarchy.")
@click.option("--max-num-nodes", required=True, type=int, help="The maximal number of nodes in the hierarchy.")
@click.option("--step",
              required=True,
              type=int,
              default=1,
              help="The step size when going from --min-num-nodes to --max-num-nodes.")
@click.option("--torchscript", is_flag=True, default=False, help="Whether the models are encoded as TorchScript.")
@click.option("--input-shape", help="shape of the given model (e.g., '3,224,224').")
@click.option("--strategy",
              "strategies",
              required=True,
              type=click.Choice(strategy_loader.get_available_strategy_names()),
              multiple=True,
              help="A strategy to benchmark (can be specified multiple times).")
@click.option("--repetitions",
              type=int,
              default=5,
              show_default=True,
              help="How often to repeat the algorithm benchmarks for each problem size.")
@click.option("--plot", is_flag=True, default=False, help="Export plot images that illustrate the results.")
@click.option("--csv", is_flag=True, default=False, help="Export the results as csv files.")
@click.option("--seed", type=int, help="The seed to use for the random number generator.")
@click.option("--is-profile", is_flag=True, help="Whether the given path points to a model profile.")
@click.argument("filename", type=click.Path(exists=True))
def vary_hierarchy_size(scenario: str, min_num_nodes: int, max_num_nodes: int, step: int, torchscript: bool,
                        input_shape: str, strategies: str, repetitions: int, plot: bool, csv: bool, seed: int,
                        is_profile: bool, filename: str) -> None:
    if seed is not None:
        random.seed(seed)

    scenario_generator = scenarios[scenario]

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

    problem_sizes = range(min_num_nodes, max_num_nodes + 1, step)

    average_durations: Dict[str, List[float]] = {strategy_name: [] for strategy_name in strategies}
    average_latencies: Dict[str, List[float]] = {strategy_name: [] for strategy_name in strategies}
    median_latencies: Dict[str, List[float]] = {strategy_name: [] for strategy_name in strategies}
    average_memory_usages: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategies}
    average_number_of_split_points: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategies}
    average_number_of_allocated_nodes: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategies}

    for problem_size in problem_sizes:
        durations: Dict[str, List[float]] = {strategy_name: [] for strategy_name in strategies}
        latencies: Dict[str, List[float]] = {strategy_name: [] for strategy_name in strategies}
        memory_usages: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategies}
        number_of_split_points: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategies}
        number_of_allocated_nodes: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategies}

        for repetition in range(repetitions):
            nodes = scenario_generator(problem_size)
            print("problem size {}".format(problem_size))

            for strategy_name in strategies:
                print("apply {} strategy".format(strategy_name))

                result_queue: multiprocessing.Queue = multiprocessing.Queue()
                executor = addnn.benchmark.strategies.executor.Executor(strategy_name, nodes, layer_profiles, seed,
                                                                        result_queue)
                executor.start()
                result = result_queue.get()
                executor.join()
                latency = estimate_ent_to_end_latency(result.placement, nodes, layer_profiles)
                current_number_of_split_points = _count_number_of_split_points(result.placement)
                current_number_of_allocated_nodes = len(set(result.placement))

                durations[strategy_name].append(result.duration)
                latencies[strategy_name].append(latency)
                memory_usages[strategy_name].append(result.memory_usage)
                number_of_split_points[strategy_name].append(current_number_of_split_points)
                number_of_allocated_nodes[strategy_name].append(current_number_of_allocated_nodes)

                print("{} solution duration: {}".format(strategy_name, result.duration))
                print("{} latency: {}".format(strategy_name, latency))
                print("{} memory: {}".format(strategy_name, result.memory_usage))
                print("{} placement: {}".format(strategy_name, result.placement))
                print("{} split points: {}".format(strategy_name, current_number_of_split_points))
                print("{} allocated nodes: {}".format(strategy_name, current_number_of_allocated_nodes))

        for strategy_name in strategies:
            average_duration = sum(durations[strategy_name]) / repetitions
            average_durations[strategy_name].append(average_duration)

            average_latency = sum(latencies[strategy_name]) / repetitions
            average_latencies[strategy_name].append(average_latency)

            median_latency = statistics.median(latencies[strategy_name])
            median_latencies[strategy_name].append(median_latency)

            average_memory_usage = sum(memory_usages[strategy_name]) / repetitions
            average_memory_usages[strategy_name].append(int(average_memory_usage))

            current_average_number_of_split_points = sum(number_of_split_points[strategy_name]) / repetitions
            average_number_of_split_points[strategy_name].append(int(current_average_number_of_split_points))

            current_average_number_of_allocated_nodes = sum(number_of_allocated_nodes[strategy_name]) / repetitions
            average_number_of_allocated_nodes[strategy_name].append(int(current_average_number_of_allocated_nodes))

    base_strategy = strategies[0]

    average_latency_losses: Dict[str, List[float]] = dict()
    for strategy in strategies:
        strategy_average_latency_losses = []
        for run_index, latency in enumerate(average_latencies[strategy]):
            optimal_latency = average_latencies[base_strategy][run_index]
            loss = latency / optimal_latency
            strategy_average_latency_losses.append(loss)
            average_latency_losses[strategy] = strategy_average_latency_losses

    median_latency_losses: Dict[str, List[float]] = dict()
    for strategy in strategies:
        strategy_median_latency_losses = []
        for run_index, latency in enumerate(median_latencies[strategy]):
            optimal_latency = median_latencies[base_strategy][run_index]
            loss = latency / optimal_latency
            strategy_median_latency_losses.append(loss)
            median_latency_losses[strategy] = strategy_median_latency_losses

    if plot:
        xlabel = "Number of Compute Nodes"
        _create_line_plot(problem_sizes, average_durations, xlabel=xlabel, ylabel="Seconds", path="execution-time.pdf")
        _create_line_plot(problem_sizes, average_latencies, xlabel=xlabel, ylabel="Seconds", path="average-latency.pdf")
        _create_line_plot(problem_sizes, median_latencies, xlabel=xlabel, ylabel="Seconds", path="median-latency.pdf")
        _create_line_plot(problem_sizes,
                          average_latency_losses,
                          xlabel=xlabel,
                          ylabel="",
                          path="average-latency-loss.pdf")
        _create_line_plot(problem_sizes,
                          median_latency_losses,
                          xlabel=xlabel,
                          ylabel="",
                          path="median-latency-loss.pdf")
        _create_line_plot(problem_sizes, average_memory_usages, xlabel=xlabel, ylabel="Bytes", path="memory.pdf")
        _create_line_plot(problem_sizes,
                          average_number_of_split_points,
                          xlabel=xlabel,
                          ylabel="Split Points",
                          path="split-points.pdf")
        _create_line_plot(problem_sizes,
                          average_number_of_allocated_nodes,
                          xlabel=xlabel,
                          ylabel="Allocated Compute Nodes",
                          path="allocated-nodes.pdf")

    if csv:
        _export_csv(problem_sizes, average_durations, "execution-time.csv")
        _export_csv(problem_sizes, average_latencies, "average-latency.csv")
        _export_csv(problem_sizes, median_latencies, "median-latency.csv")
        _export_csv(problem_sizes, average_latency_losses, "average-latency-loss.csv")
        _export_csv(problem_sizes, median_latency_losses, "median-latency-loss.csv")
        _export_csv(problem_sizes, average_memory_usages, "memory.csv")
        _export_csv(problem_sizes, average_number_of_split_points, "split-points.csv")
        _export_csv(problem_sizes, average_number_of_allocated_nodes, "allocated-nodes.csv")


@strategies.command("vary-models",
                    help="Benchmark placement algorithms on different models on a fixed compute hierarchy.")
@click.option("--scenario", required=True, type=click.Choice(scenarios.keys()), help="The benchmark scenario.")
@click.option("--num-nodes", required=True, type=int, help="The number of nodes in the compute hierarchy.")
@click.option("--torchscript", is_flag=True, default=False, help="Whether the models are encoded as TorchScript.")
@click.option("--input-shape", help="shape of the given model (e.g., '3,224,224').")
@click.option("--strategy",
              required=True,
              type=click.Choice(strategy_loader.get_available_strategy_names()),
              multiple=True,
              help="A strategy to benchmark (can be specified multiple times).")
@click.option("--repetitions",
              type=int,
              default=5,
              show_default=True,
              help="How often to repeat the algorithm benchmarks for each model size.")
@click.option("--plot", is_flag=True, default=False, help="Export plot images that illustrate the results.")
@click.option("--csv", is_flag=True, default=False, help="Export the results as csv files.")
@click.option("--seed", type=int, help="The seed to use for the random number generator.")
@click.option("--model",
              "model_names_and_paths",
              required=True,
              type=(str, click.Path(exists=True)),
              multiple=True,
              help="A model instance to benchmark (can be specified multiple times).")
def vary_models(scenario: str, num_nodes: int, torchscript: bool, input_shape: str, strategy: str, repetitions: int,
                plot: bool, csv: bool, seed: int, model_names_and_paths: List[Tuple[str, str]]) -> None:

    if seed is not None:
        random.seed(seed)

    scenario_generator = scenarios[scenario]

    models: List[addnn.model.Model] = []
    for model_name, model_path in model_names_and_paths:
        if torchscript:
            model = read_model_from_preprocessed_script(model_path, input_shape)
        else:
            model = torch.load(model_path)
        models.append(model)

    average_durations: Dict[str, List[float]] = {strategy_name: [] for strategy_name in strategy}
    average_latencies: Dict[str, List[float]] = {strategy_name: [] for strategy_name in strategy}
    average_memory_usages: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategy}
    average_number_of_split_points: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategy}

    for model in models:
        layer_profiles = get_layer_profiles(model.layers)

        durations: Dict[str, List[float]] = {strategy_name: [] for strategy_name in strategy}
        latencies: Dict[str, List[float]] = {strategy_name: [] for strategy_name in strategy}
        memory_usages: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategy}
        number_of_split_points: Dict[str, List[int]] = {strategy_name: [] for strategy_name in strategy}

        for repetition in range(repetitions):
            nodes = scenario_generator(num_nodes)

            for strategy_name in strategy:
                print("apply {} strategy".format(strategy_name))

                result_queue: multiprocessing.Queue = multiprocessing.Queue()
                executor = addnn.benchmark.strategies.executor.Executor(strategy_name, nodes, layer_profiles, seed,
                                                                        result_queue)
                executor.start()
                result = result_queue.get()
                executor.join()
                latency = estimate_ent_to_end_latency(result.placement, nodes, layer_profiles)
                current_number_of_split_points = _count_number_of_split_points(result.placement)

                durations[strategy_name].append(result.duration)
                latencies[strategy_name].append(latency)
                memory_usages[strategy_name].append(result.memory_usage)
                number_of_split_points[strategy_name].append(current_number_of_split_points)

                print("{} solution duration: {}".format(strategy_name, result.duration))
                print("{} latency: {}".format(strategy_name, latency))
                print("{} memory: {}".format(strategy_name, result.memory_usage))
                print("{} placement: {}".format(strategy_name, result.placement))
                print("{} split points: {}".format(strategy_name, current_number_of_split_points))

        for strategy_name in strategy:
            average_duration = sum(durations[strategy_name]) / repetitions
            average_durations[strategy_name].append(average_duration)

            average_latency = sum(latencies[strategy_name]) / repetitions
            average_latencies[strategy_name].append(average_latency)

            average_memory_usage = sum(memory_usages[strategy_name]) / repetitions
            average_memory_usages[strategy_name].append(int(average_memory_usage))

            current_average_number_of_split_points = sum(number_of_split_points[strategy_name]) / repetitions
            average_number_of_split_points[strategy_name].append(int(current_average_number_of_split_points))

    model_names = [model_name for model_name, model_path in model_names_and_paths]
    if plot:
        xlabel = "Model Type"
        _create_line_plot(model_names, average_durations, xlabel=xlabel, ylabel="Seconds", path="execution-time.pdf")
        _create_line_plot(model_names, average_latencies, xlabel=xlabel, ylabel="Seconds", path="latency.pdf")
        _create_line_plot(model_names, average_memory_usages, xlabel=xlabel, ylabel="Bytes", path="memory.pdf")
        _create_line_plot(model_names,
                          average_number_of_split_points,
                          xlabel=xlabel,
                          ylabel="Split Points",
                          path="split-points.pdf")

    if csv:
        _export_csv(model_names, average_durations, "execution-time.csv")
        _export_csv(model_names, average_latencies, "latency.csv")
        _export_csv(model_names, average_memory_usages, "memory.csv")
        _export_csv(model_names, average_number_of_split_points, "split-points.csv")


def _count_number_of_split_points(placement: Placement) -> int:
    current_node_index = placement[0]
    number_of_split_points = 0
    for node_index in placement:
        if current_node_index != node_index:
            current_node_index = node_index
            number_of_split_points += 1
    return number_of_split_points


def _create_line_plot(xvalues: Iterable[Any], strategy_results: Dict[str, List[Any]], xlabel: str, ylabel: str,
                      path: str) -> None:
    # clear plot
    pyplot.clf()
    pyplot.cla()

    for strategy_name in strategy_results.keys():
        pyplot.plot(list(xvalues), strategy_results[strategy_name], label=strategy_name)

    pyplot.xlabel(xlabel)
    pyplot.xticks(list(xvalues))
    pyplot.ylabel(ylabel)
    pyplot.legend()
    pyplot.savefig(path)
    print("Saved plot as {}".format(path))


def _create_bar_plot(xvalues: Iterable[Any], strategy_results: Dict[str, List[Any]], xlabel: str, ylabel: str,
                     path: str) -> None:
    # clear plot
    pyplot.clf()
    pyplot.cla()

    total_width = 0.7
    width = total_width / len(strategy_results.keys())
    xshift = total_width / 2 - width / 2

    strategy_index = 0
    for strategy_name in strategy_results.keys():
        xpositions = [xvalue - xshift + strategy_index * width for xvalue in xvalues]
        pyplot.bar(xpositions, strategy_results[strategy_name], width=width, label=strategy_name)
        strategy_index += 1

    pyplot.xlabel(xlabel)
    pyplot.xticks(list(xvalues))
    pyplot.ylabel(ylabel)
    pyplot.legend()
    pyplot.savefig(path)
    print("Saved plot as {}".format(path))


def _export_csv(xvalues: Iterable[Any], strategy_results: Dict[str, List[Any]], path: str) -> None:
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        rows = zip(xvalues, *strategy_results.values())

        strategy_names = strategy_results.keys()
        title_row = ["problem_size"] + list(strategy_names)
        csvwriter.writerow(title_row)

        for row in rows:
            csvwriter.writerow(row)
        print("Saved CSV as {}".format(path))
