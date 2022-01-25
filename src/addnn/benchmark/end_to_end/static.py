import addnn
import addnn.controller.controller
import addnn.grpc
import addnn.profile
import addnn.serve.cli
import addnn.util.controller
import click
import csv
import google.protobuf.json_format
import grpc
import multiprocessing
import os
import pickle
import random
import torch
import time
from abc import ABC, abstractmethod
from addnn.benchmark.end_to_end.cli import end_to_end
from addnn.benchmark.end_to_end.config import BenchmarkConfig, NodeConfig, SchedulerConfig, read_node_configs
from addnn.benchmark.end_to_end.remote_control.controller import *
from addnn.benchmark.end_to_end.remote_control.node import *
from addnn.benchmark.end_to_end.remote_control.scheduler import *
from addnn.benchmark.end_to_end.remote_control.host import *
from addnn.benchmark.end_to_end.request_generator import generate_requests_in_multiple_processes, InferenceBenchmark
from addnn.controller.proto.controller_pb2_grpc import ControllerStub
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from addnn.dataset import Dataset
from addnn.node.proto.neural_network_pb2 import InferRequest
from addnn.node.proto.neural_network_pb2_grpc import NeuralNetworkStub
from addnn.profile.layer_profile import LayerProfile, get_layer_profiles
from addnn.serve.placement import strategy_loader
from addnn.serve.proto import scheduler_pb2
from addnn.util.torchscript import read_model_from_preprocessed_script
from google.protobuf.empty_pb2 import Empty
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TextIO, TypeVar

STRESS_WARM_UP_DURATION = 5
WARM_UP_DURATION = 10
COOL_DOWN_DURATION = 10


class BenchmarkResult:
    def __init__(self, benchmark_config: BenchmarkConfig, inference_benchmarks: List[InferenceBenchmark],
                 placement: List[controller_pb2.RegisteredNode], scheduler_reports: List[scheduler_pb2.Report]) -> None:
        self._benchmark_config = benchmark_config
        self._inference_benchmarks = inference_benchmarks
        self._placement = placement
        self._scheduler_reports = scheduler_reports

    @property
    def inference_benchmarks(self) -> List[InferenceBenchmark]:
        return self._inference_benchmarks

    @property
    def placement(self) -> List[controller_pb2.RegisteredNode]:
        return self._placement

    @property
    def scheduler_reports(self) -> List[scheduler_pb2.Report]:
        return self._scheduler_reports

    def inference_throughput(self) -> float:
        inference_throughput = 1.0 / self.average_inference_latency()
        return inference_throughput

    def average_inference_latency(self) -> float:
        inference_latencies = [(inference_benchmark.response_time - inference_benchmark.request_time) / 1000**3
                               for inference_benchmark in self._inference_benchmarks]
        return sum(inference_latencies) / len(inference_latencies)


class BenchmarkResults:
    def __init__(self, exact_result: BenchmarkResult, cloud_result: BenchmarkResult, device_result: BenchmarkResult,
                 device_exit_result: BenchmarkResult) -> None:
        self._exact_result = exact_result
        self._cloud_result = cloud_result
        self._device_result = device_result
        self._device_exit_result = device_exit_result

    @property
    def exact_result(self) -> BenchmarkResult:
        return self._exact_result

    @property
    def cloud_result(self) -> BenchmarkResult:
        return self._cloud_result

    @property
    def device_result(self) -> BenchmarkResult:
        return self._device_result

    @property
    def device_exit_result(self) -> BenchmarkResult:
        return self._device_exit_result


@end_to_end.command(
    "server_cpu_stress",
    help="Simple scenario that involves two nodes: a single device node and a single (cloud) server node.",
    short_help="Simulates CPU stress on the server.")
@click.option("--stress-level",
              "stress_levels",
              required=True,
              type=int,
              multiple=True,
              help="The number of cores to stress.")
@click.pass_obj
def server_cpu_stress(benchmark_config: BenchmarkConfig, stress_levels: List[int]) -> None:
    cloud_node = benchmark_config.node_configs[1]
    stress_commands = [_build_cpu_stress_command(stress_level) for stress_level in stress_levels]
    _run_static_stress_benchmarks(benchmark_config, cloud_node, stress_commands, stress_levels)


def _build_cpu_stress_command(stress_level: int) -> Optional[str]:
    if stress_level == 0:
        return None
    else:
        taskset = ",".join([str(cpu_index) for cpu_index in range(stress_level)])
        return "nohup stress-ng -c {} --taskset {} &> stress-ng.log &".format(
            stress_level, taskset)  # TODO max duration to ensure that stress does not keep running by accident


@end_to_end.command("device_network_delay", help="Simulates egress network delay on the input device.")
@click.option("--delay", "delays", required=True, type=int, multiple=True, help="The CPU stress levels.")
@click.pass_obj
def device_network_delay(benchmark_config: BenchmarkConfig, delays: List[int]) -> None:
    device_node = benchmark_config.node_configs[0]
    stress_commands = [_build_network_delay_command(device_node, delay) for delay in delays]
    _run_static_stress_benchmarks(benchmark_config, device_node, stress_commands, delays)


def _build_network_delay_command(node: NodeConfig, delay: int) -> Optional[str]:
    if delay == 0:
        return None
    else:
        return "sudo tc qdisc add dev {network_device} root netem delay {delay}ms".format(
            network_device=node.network_device, delay=delay)


def _run_static_stress_benchmarks(benchmark_config: BenchmarkConfig, node_to_stress: NodeConfig,
                                  stress_commands: List[Optional[str]], stress_levels: List[int]) -> None:
    # remove existing cached model files from all nodes
    for node_config in benchmark_config.node_configs:
        execute_remote_command(node_config.host, node_config.user, "rm -rf addnn/model_cache", node_config.ssh_key_path)

    results = []
    for stress_index, stress_command in enumerate(stress_commands):
        result = _run_static_stress_benchmark(benchmark_config, node_to_stress, stress_levels[stress_index],
                                              stress_command)
        results.append(result)

    _export_inference_request_latencies(benchmark_config.result_dir, stress_levels, results)
    _export_throughput_per_stress_level(benchmark_config.result_dir, stress_levels, results)
    _export_estimated_throughput_per_stress_level(benchmark_config.result_dir, stress_levels, results)
    _export_latency_per_stress_level(benchmark_config.result_dir, stress_levels, results)
    _export_estimated_latency_per_stress_level(benchmark_config.result_dir, stress_levels, results)
    _export_network_throughput_per_stress_level(benchmark_config.result_dir, benchmark_config.node_configs,
                                                stress_levels, results)
    _export_network_latency_per_stress_level(benchmark_config.result_dir, benchmark_config.node_configs, stress_levels,
                                             results)
    _export_compute_resources_per_stress_level(benchmark_config.result_dir, benchmark_config.node_configs,
                                               stress_levels, results)


def _run_static_stress_benchmark(benchmark_config: BenchmarkConfig, node_to_stress: NodeConfig, stress_level: int,
                                 stress_command: Optional[str]) -> BenchmarkResults:
    exact_result = _run_static_stress_benchmark_for_placement_type(benchmark_config,
                                                                   node_to_stress,
                                                                   stress_level,
                                                                   stress_command,
                                                                   "optimal",
                                                                   iperf_time=5)

    device_node = benchmark_config.node_configs[0]
    cloud_node = benchmark_config.node_configs[1]

    exact_is_cloud_placement = all([node.node.host == cloud_node.host for node in exact_result.placement])
    exact_is_device_placement = all([node.node.host == device_node.host for node in exact_result.placement])

    # if the exact placement corresponds to cloud-only placement then there's no need to run the cloud-only strategy
    if exact_is_cloud_placement:
        cloud_result = exact_result
    else:
        cloud_result = _run_static_stress_benchmark_for_placement_type(benchmark_config,
                                                                       node_to_stress,
                                                                       stress_level,
                                                                       stress_command,
                                                                       "cloud",
                                                                       iperf_time=5)

    # if the exact placement corresponds to cloud-only placement then there's no need to run the device-only strategy
    if exact_is_device_placement:
        device_result = exact_result
    else:
        device_result = _run_static_stress_benchmark_for_placement_type(benchmark_config,
                                                                        node_to_stress,
                                                                        stress_level,
                                                                        stress_command,
                                                                        "ffd",
                                                                        iperf_time=5)

    device_exit_result = _run_static_stress_benchmark_for_placement_type(benchmark_config,
                                                                         node_to_stress,
                                                                         stress_level,
                                                                         stress_command,
                                                                         "device-exit",
                                                                         iperf_time=5)

    if exact_is_cloud_placement:
        print("exact placement is equal to cloud-only placement")
    elif exact_is_device_placement:
        print("exact placement is equal to device-only placement")

    results = BenchmarkResults(exact_result, cloud_result, device_result, device_exit_result)
    return results


def _run_static_stress_benchmark_for_placement_type(benchmark_config: BenchmarkConfig, node_to_stress: NodeConfig,
                                                    stress_level: int, stress_command: Optional[str],
                                                    placement_type: str, iperf_time: int) -> BenchmarkResult:
    print("run benchmark for placement: {}".format(placement_type))
    start_controller(benchmark_config)

    dataset = addnn.dataset.datasets[benchmark_config.dataset_name]

    reset_compute_nodes(benchmark_config.node_configs)

    # configure stress on the according node
    if stress_command is not None:
        print("execute stress command: ")
        print(stress_command)
        execute_remote_command(node_to_stress.host, node_to_stress.user, stress_command, node_to_stress.ssh_key_path)

        # wait a few seconds before starting the node runtimes to ensure that the stress is observable for the resource monitor on the nodes
        time.sleep(STRESS_WARM_UP_DURATION)

    initialization_start_time = time.time()

    start_compute_nodes(benchmark_config.controller_host,
                        benchmark_config.controller_port,
                        benchmark_config.node_configs,
                        use_existing_model_cache=True,
                        network_monitor_interval=10000,
                        iperf_time=iperf_time)
    wait_process = multiprocessing.Process(target=wait_until_nodes_are_initialized,
                                           args=(benchmark_config.controller_host, benchmark_config.controller_port,
                                                 benchmark_config.node_configs))
    wait_process.start()
    wait_process.join()

    initialization_end_time = time.time()
    initialization_time = initialization_end_time - initialization_start_time
    print("initialization time: {}".format(initialization_time))

    if benchmark_config.scheduler_config is not None:
        reset_scheduler(benchmark_config.scheduler_config)
        run_scheduler_once(benchmark_config.scheduler_config, benchmark_config.controller_host,
                           benchmark_config.controller_port, placement_type, dataset.input_size)

    print("wait for initial deployment to complete")
    while not is_initial_deployment_finished(benchmark_config.controller_host, benchmark_config.controller_port,
                                             benchmark_config.num_layers):
        time.sleep(1)

    placement = get_current_placement(benchmark_config.controller_host, benchmark_config.controller_port,
                                      benchmark_config.num_layers)
    if placement is None:
        raise Exception("cannot determine current placement")

    inference_benchmarks = generate_requests_in_multiple_processes(benchmark_config.controller_host,
                                                                   benchmark_config.controller_port, dataset,
                                                                   benchmark_config.dataset_root,
                                                                   benchmark_config.benchmark_duration,
                                                                   benchmark_config.seed)

    reset_compute_nodes(benchmark_config.node_configs)

    if benchmark_config.start_controller:
        if benchmark_config.controller_user is None:
            raise Exception("please provide a value for --controller-user")
        reset_remote_controller(benchmark_config.controller_host, benchmark_config.controller_user,
                                benchmark_config.controller_ssh_key)

    scheduler_reports = []

    if benchmark_config.scheduler_config is not None:
        result_dir = benchmark_config.result_dir + "/stress{}".format(stress_level)
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        scheduler_report_path = result_dir + "/" + placement_type + "-schedule0.json"
        copy_from_remote_host(benchmark_config.scheduler_config.host,
                              benchmark_config.scheduler_config.user,
                              "~/addnn/scheduler-reports/schedule0.json",
                              scheduler_report_path,
                              is_directory=False,
                              ssh_key_path=benchmark_config.scheduler_config.ssh_key_path)

        optimal_scheduler_report_path = result_dir + "/optimal-schedule0.json"
        with open(optimal_scheduler_report_path, "r") as scheduler_report_file:
            report_json = scheduler_report_file.read()
            report = google.protobuf.json_format.Parse(report_json, scheduler_pb2.Report())

            if placement_type != "optimal":
                # compute the estimated latency for the placement decision of the benchmarked strategy, based on the
                # environmental conditions that were recorded at the begin of benchmark-run for the exact/optimal
                # strategy
                nodes = [registered_node.node for registered_node in report.nodes]
                layer_profiles = [_parse_layer_profile(layer_profile) for layer_profile in report.layer_profiles]
                current_placement = addnn.serve.scheduler._compute_placement(placement_type, nodes, layer_profiles)
                report.placement.nodes.extend([report.nodes[node_index].uuid for node_index in current_placement])
                report.estimated_latency = addnn.serve.placement.placement.estimate_ent_to_end_latency(
                    current_placement, nodes, layer_profiles)

            scheduler_reports.append(report)

    result = BenchmarkResult(benchmark_config, inference_benchmarks, placement, scheduler_reports)
    return result


def _parse_layer_profile(pb_layer_profile: scheduler_pb2.LayerProfile) -> LayerProfile:
    return LayerProfile(flops=pb_layer_profile.flops,
                        in_memory_size=pb_layer_profile.in_memory_size,
                        storage_size=pb_layer_profile.storage_size,
                        marshalled_input_size=pb_layer_profile.marshalled_input_size,
                        has_exit=pb_layer_profile.has_exit,
                        number_of_exited_samples=pb_layer_profile.number_of_exited_samples,
                        exit_probability=pb_layer_profile.exit_probability,
                        execution_probability=pb_layer_profile.execution_probability,
                        operators=list(pb_layer_profile.operators))


def _export_inference_request_latencies(result_dir: str, stress_levels: List[int],
                                        results: List[BenchmarkResults]) -> None:
    for stress_index, stress_level in enumerate(stress_levels):
        stress_result_dir = result_dir + "/stress{}".format(stress_level)
        _export_inference_request_latencies_for_placement_type(stress_result_dir, "exact",
                                                               results[stress_index].exact_result)
        _export_inference_request_latencies_for_placement_type(stress_result_dir, "cloud",
                                                               results[stress_index].cloud_result)
        _export_inference_request_latencies_for_placement_type(stress_result_dir, "device",
                                                               results[stress_index].device_result)
        _export_inference_request_latencies_for_placement_type(stress_result_dir, "device-exit",
                                                               results[stress_index].device_exit_result)


def _export_inference_request_latencies_for_placement_type(result_dir: str, placement_type: str,
                                                           result: BenchmarkResult) -> None:
    result_filename = result_dir + "/" + placement_type + "-requests.csv"
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        title_row = ["request_unix_time", "response_unix_time", "latency_seconds"]
        csvwriter.writerow(title_row)

        for inference_benchmark in result.inference_benchmarks:
            latency = (inference_benchmark.response_time - inference_benchmark.request_time) / 1000**3
            row = [inference_benchmark.request_time, inference_benchmark.response_time, latency]
            csvwriter.writerow(row)


def _export_throughput_per_stress_level(result_dir: str, stress_levels: List[int],
                                        results: List[BenchmarkResults]) -> None:
    result_filename = result_dir + "/actual_throughput.csv"
    _export_result_per_stress_level(result_filename,
                                    stress_levels,
                                    results,
                                    transform=lambda result: result.inference_throughput())


def _export_latency_per_stress_level(result_dir: str, stress_levels: List[int],
                                     results: List[BenchmarkResults]) -> None:
    result_filename = result_dir + "/actual_latency.csv"
    _export_result_per_stress_level(result_filename,
                                    stress_levels,
                                    results,
                                    transform=lambda result: result.average_inference_latency())


def _export_estimated_latency_per_stress_level(result_dir: str, stress_levels: List[int],
                                               results: List[BenchmarkResults]) -> None:
    result_filename = result_dir + "/estimated_latency.csv"
    _export_result_per_stress_level(result_filename,
                                    stress_levels,
                                    results,
                                    transform=lambda result: result.scheduler_reports[0].estimated_latency)


def _export_estimated_throughput_per_stress_level(result_dir: str, stress_levels: List[int],
                                                  results: List[BenchmarkResults]) -> None:
    result_filename = result_dir + "/estimated_throughput.csv"
    _export_result_per_stress_level(result_filename,
                                    stress_levels,
                                    results,
                                    transform=lambda result: 1.0 / result.scheduler_reports[0].estimated_latency)


def _export_network_throughput_per_stress_level(result_dir: str, node_configs: List[NodeConfig],
                                                stress_levels: List[int], results: List[BenchmarkResults]) -> None:
    device_node = node_configs[0]
    result_filename = result_dir + "/device2cloud_network_throughput.csv"
    _export_result_per_stress_level(
        result_filename,
        stress_levels,
        results,
        transform=lambda result: _get_network_throughput(result.scheduler_reports[0], device_node.host))

    cloud_node = node_configs[1]
    result_filename = result_dir + "/cloud2device_network_throughput.csv"
    _export_result_per_stress_level(
        result_filename,
        stress_levels,
        results,
        transform=lambda result: _get_network_throughput(result.scheduler_reports[0], cloud_node.host))


def _get_network_throughput(scheduler_report: scheduler_pb2.Report, from_host: str) -> float:
    from_node = _get_node_by_host(scheduler_report.nodes, from_host)
    network_throughput = from_node.node.state.resource_state.network_throughputs[0]
    return network_throughput.throughput / 1000 / 1000


def _export_network_latency_per_stress_level(result_dir: str, node_configs: List[NodeConfig], stress_levels: List[int],
                                             results: List[BenchmarkResults]) -> None:
    device_node = node_configs[0]
    result_filename = result_dir + "/device2cloud_network_latency.csv"
    _export_result_per_stress_level(
        result_filename,
        stress_levels,
        results,
        transform=lambda result: _get_network_latency(result.scheduler_reports[0], device_node.host))

    cloud_node = node_configs[1]
    result_filename = result_dir + "/cloud2device_network_latency.csv"
    _export_result_per_stress_level(
        result_filename,
        stress_levels,
        results,
        transform=lambda result: _get_network_latency(result.scheduler_reports[0], cloud_node.host))


def _export_compute_resources_per_stress_level(result_dir: str, node_configs: List[NodeConfig],
                                               stress_levels: List[int], results: List[BenchmarkResults]) -> None:
    device_node = node_configs[0]
    result_filename = result_dir + "/device_gflops.csv"
    _export_result_per_stress_level(result_filename,
                                    stress_levels,
                                    results,
                                    transform=lambda result: _get_node_by_host(result.scheduler_reports[
                                        0].nodes, device_node.host).node.state.resource_state.compute / 1000**3)

    cloud_node = node_configs[1]
    result_filename = result_dir + "/cloud_gflops.csv"
    _export_result_per_stress_level(result_filename,
                                    stress_levels,
                                    results,
                                    transform=lambda result: _get_node_by_host(result.scheduler_reports[
                                        0].nodes, cloud_node.host).node.state.resource_state.compute / 1000**3)


def _get_network_latency(scheduler_report: scheduler_pb2.Report, from_host: str) -> float:
    from_node = _get_node_by_host(scheduler_report.nodes, from_host)
    network_latency = from_node.node.state.resource_state.network_latencies[0]
    return network_latency.latency


def _export_result_per_stress_level(result_filename: str, stress_levels: List[int], results: List[BenchmarkResults],
                                    transform: Callable[[BenchmarkResult], float]) -> None:
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        exact = [transform(result.exact_result) for result in results]
        cloud = [transform(result.cloud_result) for result in results]
        device = [transform(result.device_result) for result in results]
        device_exit = [transform(result.device_exit_result) for result in results]

        rows = zip(stress_levels, exact, cloud, device, device_exit)

        title_row = ["stress", "exact", "cloud", "device", "device-exit"]
        csvwriter.writerow(title_row)

        for row in rows:
            csvwriter.writerow(row)


def _get_node_by_host(nodes: Iterable[controller_pb2.RegisteredNode], host: str) -> controller_pb2.RegisteredNode:
    node = list(filter(lambda registered_node: registered_node.node.host == host, nodes))
    return node[0]
