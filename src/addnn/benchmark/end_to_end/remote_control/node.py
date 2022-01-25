import addnn
import addnn.serve.scheduler
import grpc
import multiprocessing
import subprocess
import time
from addnn.benchmark.end_to_end.config import BenchmarkConfig, NodeConfig, SchedulerConfig, read_node_configs
from addnn.benchmark.end_to_end.remote_control.host import *
from addnn.controller.proto.controller_pb2_grpc import ControllerStub
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from google.protobuf.empty_pb2 import Empty
from typing import Dict, Iterable, List, Optional


def reset_compute_nodes(node_configs: List[NodeConfig]) -> None:
    for node_config in node_configs:
        _reset_compute_node(node_config)


def _reset_compute_node(node_config: NodeConfig) -> None:
    print("reset compute node at {}".format(node_config.host))

    command = "cat ~/node.pid 2> /dev/null | xargs kill &> /dev/null ; rm -f ~/node.pid ; killall -q stress-ng ; sudo tc qdisc del dev {} root &> /dev/null ; true".format(
        node_config.network_device)
    execute_remote_command(node_config.host, node_config.user, command, node_config.ssh_key_path)


def start_compute_nodes(controller_host: str, controller_port: int, node_configs: List[NodeConfig],
                        use_existing_model_cache: bool, network_monitor_interval: int, iperf_time: int) -> None:
    for node_config in node_configs:
        _start_compute_node(controller_host, controller_port, node_config, use_existing_model_cache,
                            network_monitor_interval, iperf_time)


def _start_compute_node(controller_host: str, controller_port: int, node_config: NodeConfig,
                        use_existing_model_cache: bool, network_monitor_interval: int, iperf_time: int) -> None:
    print("start compute node at {}".format(node_config.host))

    command = "cd addnn ; "
    command += "nohup {} node ".format(node_config.addnn_executable)
    command += "--bind-ip={} ".format(node_config.host)
    command += "--bind-port={} ".format(node_config.port)
    command += "--controller-host={} ".format(controller_host)
    command += "--controller-port={} ".format(controller_port)
    command += "--compute={} ".format(node_config.compute_capacity)
    command += "--tier={} ".format(node_config.tier)
    if use_existing_model_cache:
        command += "--use-existing-model-cache "
    command += "--network-monitor-interval={} ".format(network_monitor_interval)
    command += "--iperf-time={} ".format(iperf_time)
    if node_config.is_input:
        command += "--is-input "
    command += "--iperf &> /home/ubuntu/node-log.txt & echo $! > ~/node.pid"
    execute_remote_command(node_config.host, node_config.user, command, node_config.ssh_key_path)


def wait_until_nodes_are_initialized(controller_host: str, controller_port: int,
                                     node_configs: List[NodeConfig]) -> None:
    print("wait for compute nodes to finish setup")

    controller_stub = addnn.grpc.create_tcp_stub(ControllerStub, controller_host, controller_port)

    while not _are_nodes_initialized(controller_stub, node_configs):
        time.sleep(1)

    list_nodes_response = controller_stub.ListNodes(Empty())
    print("resource states before benchmark:")
    for node in list_nodes_response.nodes:
        print("resource state of [{}]:".format(node.node.host))
        print("GFLOPS: {}".format(float(node.node.state.resource_state.compute) / 1000**3))
        for network_throughput in node.node.state.resource_state.network_throughputs:
            print("network throughput to [{}]: {} MBit/s".format(network_throughput.host,
                                                                 float(network_throughput.throughput) / 1000**2))
        for network_latency in node.node.state.resource_state.network_latencies:
            print("network latency to [{}]: {} ms".format(network_latency.host, float(network_latency.latency)))


def _are_nodes_initialized(controller_stub: ControllerStub, node_configs: List[NodeConfig]) -> bool:
    list_nodes_response = controller_stub.ListNodes(Empty())

    if len(list_nodes_response.nodes) != len(node_configs):
        return False

    number_of_neighbours = len(node_configs) - 1

    for registered_node in list_nodes_response.nodes:
        if not _is_node_initialized(registered_node.node, number_of_neighbours):
            return False

    return True


def _is_node_initialized(node: controller_pb2.Node, number_of_neighbours: int) -> bool:
    if node.state.resource_state.memory <= 0:
        return False

    if node.state.resource_state.storage <= 0:
        return False

    if node.state.resource_state.compute <= 0:
        return False

    if len(node.state.resource_state.network_throughputs) < number_of_neighbours:
        return False

    if len(node.state.resource_state.network_latencies) < number_of_neighbours:
        return False

    return True
