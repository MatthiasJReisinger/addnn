import addnn
import addnn.model
import addnn.serve.cli
import click
import grpc
import pickle
import pulp
import random
import torch
import time
from addnn.cli import cli
from addnn.controller.proto import controller_pb2
from addnn.example.models import resnet, mobilenetv3, vgg
from addnn.node.proto import node_pb2, node_state_pb2
from addnn.profile.layer_profile import LayerProfile, get_layer_profiles
from addnn.serve.placement.placement import Placement, estimate_ent_to_end_latency
from addnn.util.torchscript import read_model_from_preprocessed_script
from matplotlib import pyplot
from typing import Iterable, List, TextIO


def scenario_iot(problem_size: int) -> List[controller_pb2.Node]:
    # Simulates a cluster of Raspberry Pi A+ nodes
    iot_nodes = _create_tier(tier=0,
                             number_of_nodes=problem_size,
                             max_memory=256_000_000,
                             max_storage=500_000_000,
                             max_compute=218_000_000,
                             host_prefix="iot",
                             intra_bandwidth=5_000_000,
                             vary_intra_bandwidth=True)
    iot_nodes[0].is_input = True
    return iot_nodes


def scenario_iot_edge_cloud(problem_size: int) -> List[controller_pb2.Node]:
    number_of_tiers = 3

    # IoT Tier: Raspberry Pi A+
    number_of_iot_nodes = int(problem_size / number_of_tiers)
    iot_nodes = _create_tier(tier=0,
                             number_of_nodes=number_of_iot_nodes,
                             max_memory=256_000_000,
                             max_storage=1_000_000_000,
                             max_compute=218_000_000,
                             host_prefix="iot",
                             intra_bandwidth=1_000_000_000)
    iot_nodes[0].is_input = True

    # Edge Tier: Raspberry Pi 4B
    number_of_edge_nodes = int(problem_size / number_of_tiers)
    edge_nodes = _create_tier(tier=1,
                              number_of_nodes=number_of_edge_nodes,
                              max_memory=4_000_000_000,
                              max_storage=4_000_000_000,
                              max_compute=10_300_000_000,
                              host_prefix="edge",
                              intra_bandwidth=1_000_000_000)

    # Cloud tier: Broadwell e5 2620 v4
    number_of_cloud_nodes = problem_size - 2 * int(problem_size / number_of_tiers)
    cloud_nodes = _create_tier(tier=2,
                               number_of_nodes=number_of_cloud_nodes,
                               max_memory=32_000_000_000,
                               max_storage=100_000_000_000,
                               max_compute=184_000_000_000,
                               host_prefix="cloud",
                               intra_bandwidth=1_000_000_000)

    print("number of nodes: {}".format(number_of_iot_nodes + number_of_edge_nodes + number_of_cloud_nodes))

    _connect_tiers(iot_nodes, edge_nodes, throughput=50_000_000, latency=25)
    _connect_tiers(iot_nodes, cloud_nodes, throughput=5_000_000, latency=50)
    _connect_tiers(edge_nodes, cloud_nodes, throughput=100_000_000, latency=25)

    nodes = iot_nodes + edge_nodes + cloud_nodes
    return nodes


def _scenario_iot_cloud(model: addnn.model.Model) -> None:
    stress_levels = [0.0, 0.22, 0.45, 0.67, 0.9]  # percentage of CPU & memory stress
    bandwidth_levels = [10, 25, 37.5, 50]  # Mb/s

    with open("benchmark.txt", "w") as log_file:
        benchmark_id = 0
        for cpu_level in stress_levels:
            for memory_level in stress_levels:
                for bandwidth_level in bandwidth_levels:
                    _scenario_iot_cloud_run(log_file, benchmark_id, model, cpu_level, memory_level, bandwidth_level)
                    benchmark_id += 1


def _scenario_iot_cloud_run(log_file: TextIO, benchmark_id: int, model: addnn.model.Model, cpu_stress_level: float,
                            memory_stress_level: float, bandwidth_level: float) -> None:
    model.eval()

    nodes = []

    edge_cpu_peak_performance = 6_000_000_000  # ARM Cortex-A72, single core peak performance

    edge_node = controller_pb2.Node()
    edge_node.host = "1.1.1.1"
    edge_node.port = 42
    edge_node.tier = 0
    edge_node.is_input = True
    edge_node.state.resource_state.memory = int(2_000_000_000 * (1 - memory_stress_level))
    edge_node.state.resource_state.storage = 1_000_000_000
    edge_node.state.resource_state.compute = int(edge_cpu_peak_performance * (1 - cpu_stress_level))
    edge_node.state.resource_state.bandwidth = int(bandwidth_level * 1_000_000)
    nodes.append(edge_node)

    cloud_node = controller_pb2.Node()
    cloud_node.host = "2.2.2.2"
    cloud_node.port = 42
    cloud_node.tier = 2
    cloud_node.is_input = False
    cloud_node.state.resource_state.memory = 8_000_000_000
    cloud_node.state.resource_state.storage = 8_000_000_000
    cloud_node.state.resource_state.compute = 150_000_000_000  # Intel Xeon E5
    cloud_node.state.resource_state.bandwidth = 1_000_000
    nodes.append(cloud_node)

    layers = model.layers
    layer_profiles = get_layer_profiles(layers)

    start_time = time.time()
    strategy = addnn.serve.placement.strategies.optimal.OptimalStrategy()
    layer_placement = strategy.compute_placement(nodes, layer_profiles)

    end_time = time.time()
    duration = end_time - start_time
    optimal_latency = estimate_ent_to_end_latency(layer_placement, nodes, layer_profiles)

    layers_per_node: List[List[int]] = [[] for node_index in range(len(nodes))]
    for layer_index in range(len(layers)):
        node_index = layer_placement[layer_index]
        layers_per_node[node_index].append(layer_index)

    log_file.write("benchmark {}\n".format(benchmark_id))
    log_file.write("cpu stress level:    {}\n".format(cpu_stress_level))
    log_file.write("memory stress level: {}\n".format(memory_stress_level))
    log_file.write("bandwidth:           {}\n".format(bandwidth_level))
    log_file.write("run-time:            {}\n".format(duration))
    print("inference time: {}".format(optimal_latency))

    for node_index in range(len(nodes)):
        log_file.write("node {}: {}\n".format(node_index, layers_per_node[node_index]))

    log_file.write("\n")
    log_file.flush()


def _create_tier(tier: int,
                 number_of_nodes: int,
                 max_memory: int,
                 max_storage: int,
                 max_compute: int,
                 host_prefix: str,
                 intra_bandwidth: int,
                 vary_intra_bandwidth: bool = False) -> List[controller_pb2.Node]:
    nodes = []
    for node_index in range(number_of_nodes):
        node = _create_node(tier, max_memory, max_storage, max_compute, "{}{}".format(host_prefix, node_index))
        nodes.append(node)
        node_index += 1

    for node in nodes:
        for neighbour_node in nodes:
            if node != neighbour_node:
                # initialize intra-tier throughputs
                network_throughput = node_state_pb2.NetworkThroughput()
                network_throughput.host = neighbour_node.host
                if vary_intra_bandwidth:
                    network_throughput.throughput = random.randint(int(0.1 * intra_bandwidth), intra_bandwidth)
                else:
                    network_throughput.throughput = intra_bandwidth
                node.state.resource_state.network_throughputs.append(network_throughput)

                # initialize intra-tier latencies
                network_latency = node_state_pb2.NetworkLatency()
                network_latency.host = neighbour_node.host
                network_latency.latency = 0
                node.state.resource_state.network_latencies.append(network_latency)

    return nodes


def _create_node(tier: int, max_memory: int, max_storage: int, max_compute: int, host: str) -> controller_pb2.Node:
    node = controller_pb2.Node()
    node.host = host
    node.port = 42
    node.tier = tier
    node.is_input = False
    node.state.resource_state.memory = random.randint(int(0.1 * max_memory), max_memory)
    node.state.resource_state.storage = random.randint(int(0.1 * max_storage), max_storage)
    node.state.resource_state.compute = random.randint(int(0.1 * max_compute), max_compute)
    node.state.resource_state.bandwidth = 8
    return node


def _connect_tiers(tierA: Iterable[controller_pb2.Node], tierB: Iterable[controller_pb2.Node], throughput: int,
                   latency: int) -> None:
    for nodeA in tierA:
        for nodeB in tierB:
            # initialize network throughputs between tierA and tierB
            network_throughput = node_state_pb2.NetworkThroughput()
            network_throughput.throughput = random.randint(int(0.1 * throughput), throughput)
            network_throughput.host = nodeB.host
            nodeA.state.resource_state.network_throughputs.append(network_throughput)
            network_throughput.host = nodeA.host
            nodeB.state.resource_state.network_throughputs.append(network_throughput)

            # initialize network latencies between tierA and tierB
            network_latency = node_state_pb2.NetworkLatency()
            network_latency.latency = latency
            network_latency.host = nodeB.host
            nodeA.state.resource_state.network_latencies.append(network_latency)
            network_latency.host = nodeA.host
            nodeB.state.resource_state.network_latencies.append(network_latency)
