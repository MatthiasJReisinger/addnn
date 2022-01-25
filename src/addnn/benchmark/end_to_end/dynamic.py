import addnn
import click
import csv
import google.protobuf.json_format
import multiprocessing
import os
import random
import time
import torch
from addnn.benchmark.cli import benchmark
from addnn.benchmark.end_to_end.cli import end_to_end
from addnn.benchmark.end_to_end.request_generator import generate_requests_until_stoped, InferenceBenchmark
from addnn.benchmark.end_to_end.config import BenchmarkConfig, NodeConfig, SchedulerConfig, read_node_configs
from addnn.benchmark.end_to_end.remote_control.controller import *
from addnn.benchmark.end_to_end.remote_control.node import *
from addnn.benchmark.end_to_end.remote_control.scheduler import *
from addnn.benchmark.end_to_end.remote_control.host import *
from addnn.benchmark.end_to_end.static import _get_node_by_host
from addnn.controller.proto import controller_pb2
from addnn.serve.placement import strategy_loader
from addnn.serve.proto import scheduler_pb2
from typing import Optional, List

WARM_UP_DURATION = 10


class BenchmarkResult:
    def __init__(self, benchmark_config: BenchmarkConfig, inference_benchmarks: List[InferenceBenchmark],
                 scheduler_reports: List[scheduler_pb2.Report]) -> None:
        self._benchmark_config = benchmark_config
        self._inference_benchmarks = inference_benchmarks
        self._scheduler_reports = scheduler_reports

    @property
    def inference_benchmarks(self) -> List[InferenceBenchmark]:
        return self._inference_benchmarks

    @property
    def scheduler_reports(self) -> List[scheduler_pb2.Report]:
        return self._scheduler_reports


@end_to_end.command("dynamic", help="Simple dynamic scenario that involves multiple nodes.")
@click.option("--placement",
              "placement_type",
              required=True,
              type=click.Choice(strategy_loader.get_available_strategy_names()),
              help="A placement strategy to benchmark.")
@click.option("--delay", "delays", required=True, type=int, multiple=True, help="The CPU stress levels.")
@click.option("--epoch-duration", type=int, required=True, help="The duration of each stress epoch.")
@click.pass_obj
def dynamic(benchmark_config: BenchmarkConfig, placement_type: str, delays: List[int], epoch_duration: int) -> None:
    device_node = benchmark_config.node_configs[0]
    cloud_node = benchmark_config.node_configs[1]

    start_controller(benchmark_config)

    dataset = addnn.dataset.datasets[benchmark_config.dataset_name]

    reset_compute_nodes(benchmark_config.node_configs)
    start_compute_nodes(benchmark_config.controller_host,
                        benchmark_config.controller_port,
                        benchmark_config.node_configs,
                        use_existing_model_cache=False,
                        network_monitor_interval=10,
                        iperf_time=5)
    wait_process = multiprocessing.Process(target=wait_until_nodes_are_initialized,
                                           args=(benchmark_config.controller_host, benchmark_config.controller_port,
                                                 benchmark_config.node_configs))
    wait_process.start()
    wait_process.join()

    if benchmark_config.scheduler_config is not None:
        reset_scheduler(benchmark_config.scheduler_config)
        repeat_interval = 5
        start_scheduler(benchmark_config.scheduler_config, repeat_interval, benchmark_config.controller_host,
                        benchmark_config.controller_port, placement_type, dataset.input_size)

    print("wait for initial deployment to complete")
    while not is_initial_deployment_finished(benchmark_config.controller_host, benchmark_config.controller_port,
                                             benchmark_config.num_layers):
        time.sleep(1)

    benchmark_start_event = multiprocessing.Event()
    benchmark_stop_event = multiprocessing.Event()
    result_queue: multiprocessing.Queue = multiprocessing.Queue()

    benchmark_process = multiprocessing.Process(
        target=generate_requests_until_stoped,
        args=(result_queue, benchmark_start_event, benchmark_stop_event, benchmark_config.controller_host,
              benchmark_config.controller_port, dataset, benchmark_config.dataset_root))
    benchmark_process.start()

    time.sleep(WARM_UP_DURATION)
    benchmark_start_event.set()

    # apply the epochs
    epoch_timestamps = []
    for epoch_index, delay in enumerate(delays):
        print("start epoch {}".format(epoch_index))
        _apply_delay(device_node, delay)
        epoch_timestamps.append(time.time_ns())
        time.sleep(epoch_duration)

    benchmark_stop_event.set()
    benchmark_process.join()
    inference_benchmarks: List[InferenceBenchmark] = result_queue.get()

    if benchmark_config.scheduler_config is not None:
        reset_scheduler(benchmark_config.scheduler_config)
        copy_from_remote_host(benchmark_config.scheduler_config.host,
                              benchmark_config.scheduler_config.user,
                              "~/addnn/scheduler-reports",
                              benchmark_config.result_dir,
                              is_directory=True,
                              ssh_key_path=benchmark_config.scheduler_config.ssh_key_path)

    reset_compute_nodes(benchmark_config.node_configs)

    if benchmark_config.start_controller:
        if benchmark_config.controller_user is None:
            raise Exception("please provide a value for --controller-user")
        reset_remote_controller(benchmark_config.controller_host, benchmark_config.controller_user,
                                benchmark_config.controller_ssh_key)

    benchmark_start_time = inference_benchmarks[0].request_time

    scheduler_reports = _parse_scheduler_reports(benchmark_config.result_dir)
    _export_epoch_timestamps(benchmark_config.result_dir, epoch_timestamps, benchmark_start_time)
    _export_inference_benchmarks(benchmark_config.result_dir, placement_type, inference_benchmarks)
    _export_estimated_latency(benchmark_config.result_dir, placement_type, scheduler_reports, benchmark_start_time)
    _export_split_points(benchmark_config.result_dir, placement_type, scheduler_reports, benchmark_start_time,
                         device_node, cloud_node)
    _export_device2cloud_network_throughput(benchmark_config.result_dir, scheduler_reports, device_node.host,
                                            benchmark_start_time)
    _export_device2cloud_network_latency(benchmark_config.result_dir, scheduler_reports, device_node.host,
                                         benchmark_start_time)


def _apply_delay(node: NodeConfig, delay: int) -> None:
    stress_command = "sudo tc qdisc del dev {} root &> /dev/null ; true".format(node.network_device)

    if delay > 0:
        stress_command += " ; sudo tc qdisc add dev {} root netem delay {}ms".format(node.network_device, delay)

    execute_remote_command(node.host, node.user, stress_command, node.ssh_key_path)


def _parse_scheduler_reports(result_dir: str) -> List[scheduler_pb2.Report]:
    reports = []
    scheduler_report_index = 0
    while True:
        scheduler_report_path = "{}/scheduler-reports/schedule{}.json".format(result_dir, scheduler_report_index)
        if not os.path.isfile(scheduler_report_path):
            break

        with open(scheduler_report_path, "r") as scheduler_report_file:
            report_json = scheduler_report_file.read()
            report = google.protobuf.json_format.Parse(report_json, scheduler_pb2.Report())
            reports.append(report)

        scheduler_report_index += 1

    return reports


def _export_epoch_timestamps(result_dir: str, epoch_timestamps: List[int], benchmark_start_time: int) -> None:
    result_filename = result_dir + "/epochs.csv"
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        title_row = ["relative_time"]
        csvwriter.writerow(title_row)

        for epoch_timestamp in epoch_timestamps:
            relative_time = float(epoch_timestamp - benchmark_start_time) / 1000**3
            row = [relative_time]
            csvwriter.writerow(row)


def _export_inference_benchmarks(result_dir: str, placement_type: str,
                                 inference_benchmarks: List[InferenceBenchmark]) -> None:
    result_filename = result_dir + "/" + placement_type + "-requests.csv"
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        title_row = [
            "request_unix_time", "response_unix_time", "relative_request_time", "relative_response_time", "latency"
        ]
        csvwriter.writerow(title_row)

        first_request_time = inference_benchmarks[0].request_time

        for inference_benchmark in inference_benchmarks:
            latency = float(inference_benchmark.response_time - inference_benchmark.request_time) / 1000**3
            relative_request_time = float(inference_benchmark.request_time - first_request_time) / 1000**3
            relative_response_time = float(inference_benchmark.response_time - first_request_time) / 1000**3
            row = [
                inference_benchmark.request_time, inference_benchmark.response_time, relative_request_time,
                relative_response_time, latency
            ]
            csvwriter.writerow(row)


def _export_estimated_latency(result_dir: str, placement_type: str, scheduler_reports: List[scheduler_pb2.Report],
                              benchmark_start_time: int) -> None:
    result_filename = result_dir + "/" + placement_type + "-estimated-latency.csv"
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        title_row = ["relative_schedule_time", "latency"]
        csvwriter.writerow(title_row)

        for scheduler_report in scheduler_reports:
            relative_schedule_time = (scheduler_report.end_timestamp / 1000**2) - (benchmark_start_time / 1000**3)
            row = [relative_schedule_time, scheduler_report.estimated_latency]
            csvwriter.writerow(row)


def _export_split_points(result_dir: str, placement_type: str, scheduler_reports: List[scheduler_pb2.Report],
                         benchmark_start_time: int, device_node: NodeConfig, cloud_node: NodeConfig) -> None:
    initial_schedule = scheduler_reports[0]
    device_uuid = _get_node_by_host(initial_schedule.nodes, device_node.host).uuid
    cloud_uuid = _get_node_by_host(initial_schedule.nodes, cloud_node.host).uuid

    result_filename = result_dir + "/" + placement_type + "-split-points.csv"
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        title_row = ["relative_schedule_time", "split_point", "type"]
        csvwriter.writerow(title_row)

        for scheduler_report in scheduler_reports:
            relative_schedule_time = (scheduler_report.end_timestamp / 1000**2) - (benchmark_start_time / 1000**3)
            split_point = 0
            for node_uuid in scheduler_report.placement.nodes:
                if node_uuid != device_uuid:
                    break
                split_point += 1

            # make it easier to spot in the csv if this is a "real" split point, or if all layers are assigned to a single node
            if split_point == 0:
                split_type = "cloud"
            elif split_point == len(scheduler_report.placement.nodes):
                split_type = "device"
            else:
                split_type = "split"

            row = [relative_schedule_time, split_point, split_type]
            csvwriter.writerow(row)


def _export_device2cloud_network_throughput(result_dir: str, scheduler_reports: List[scheduler_pb2.Report],
                                            device_host: str, benchmark_start_time: int) -> None:
    result_filename = result_dir + "/device2cloud-network-throughput.csv"
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        title_row = ["relative_schedule_time", "throughput"]
        csvwriter.writerow(title_row)

        for scheduler_report in scheduler_reports:
            relative_schedule_time = (scheduler_report.end_timestamp / 1000**2) - (benchmark_start_time / 1000**3)
            network_throughput = addnn.benchmark.end_to_end.static._get_network_throughput(
                scheduler_report, device_host)
            row = [relative_schedule_time, network_throughput]
            csvwriter.writerow(row)


def _export_device2cloud_network_latency(result_dir: str, scheduler_reports: List[scheduler_pb2.Report],
                                         device_host: str, benchmark_start_time: int) -> None:
    result_filename = result_dir + "/device2cloud-network-latency.csv"
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        title_row = ["relative_schedule_time", "latency"]
        csvwriter.writerow(title_row)

        for scheduler_report in scheduler_reports:
            relative_schedule_time = (scheduler_report.end_timestamp / 1000**2) - (benchmark_start_time / 1000**3)
            network_latency = addnn.benchmark.end_to_end.static._get_network_latency(scheduler_report, device_host)
            row = [relative_schedule_time, network_latency]
            csvwriter.writerow(row)


@benchmark.command("dynamic-placement-compare", help="Helper command.")
@click.option("--placement",
              "placement_type",
              required=True,
              type=click.Choice(strategy_loader.get_available_strategy_names()),
              help="A placement strategy to benchmark.")
@click.option("--result-dir",
              default="./benchmark-results",
              show_default=True,
              type=click.Path(),
              help="Path to the directory where benchmark results should be written to.")
@click.option("--relative-schedule-times-path", required=True, type=click.Path(exists=True))
@click.option("--model-path", required=True, type=click.Path(exists=True), help="Path to the model.")
def dynamic_placement_compare(placement_type: str, result_dir: str, relative_schedule_times_path: str,
                              model_path: str) -> None:
    model = torch.load(model_path)
    model.eval()
    layer_profiles = addnn.profile.layer_profile.get_layer_profiles(model.layers)

    scheduler_reports = _parse_scheduler_reports(result_dir)
    result_filename = result_dir + "/" + placement_type + "-estimated-latency.csv"

    relative_schedule_times = []
    with open(relative_schedule_times_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)  # skip title row
        for row in csvreader:
            relative_schedule_times.append(row[0])

    for scheduler_report in scheduler_reports:
        nodes = [registered_node.node for registered_node in scheduler_report.nodes]
        placement = addnn.serve.scheduler._compute_placement(placement_type, nodes, layer_profiles)
        scheduler_report.estimated_latency = addnn.serve.placement.placement.estimate_ent_to_end_latency(
            placement, nodes, layer_profiles)

    result_filename = result_dir + "/" + placement_type + "-estimated-latency.csv"
    with open(result_filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        title_row = ["relative_schedule_time", "latency"]
        csvwriter.writerow(title_row)

        for schedule_index, scheduler_report in enumerate(scheduler_reports):
            relative_schedule_time = relative_schedule_times[schedule_index]
            row = [relative_schedule_time, str(scheduler_report.estimated_latency)]
            csvwriter.writerow(row)
