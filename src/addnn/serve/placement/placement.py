import numpy as np
from addnn.controller.proto import controller_pb2
from addnn.profile.layer_profile import LayerProfile
from typing import Dict, List, Optional

NodeIndex = int

# A placement is a mapping from layer indices to node indices.
Placement = List[NodeIndex]


def estimate_ent_to_end_latency(placement: Placement,
                                nodes: List[controller_pb2.Node],
                                layers: List[LayerProfile],
                                throughput_matrix: Optional[List[Dict[str, int]]] = None,
                                latency_matrix: Optional[List[Dict[str, float]]] = None) -> float:
    if throughput_matrix is None:
        throughput_matrix = get_throughput_matrix(nodes)

    if latency_matrix is None:
        latency_matrix = get_latency_matrix(nodes)

    compute_latency = float(
        np.sum([
            layers[layer_index].execution_probability * float(layers[layer_index].flops) /
            float(nodes[int(placement[layer_index])].state.resource_state.compute) for layer_index in range(len(layers))
        ]))

    communication_latency = 0.0
    for layer_index in range(1, len(layers)):
        if placement[layer_index - 1] != placement[layer_index]:
            communication_latency += layers[layer_index].execution_probability * (
                float(layers[layer_index].marshalled_input_size) * 8 /
                float(throughput_matrix[int(placement[layer_index - 1])][nodes[int(placement[layer_index])].host]) +
                latency_matrix[int(placement[layer_index - 1])][nodes[int(placement[layer_index])].host] / 1000.0)

    input_node_index = 0
    for node_index in range(len(nodes)):
        if nodes[node_index].is_input:
            input_node_index = node_index
            break

    input_latency = 0.0
    if int(placement[0]) != input_node_index:
        input_latency = float(layers[0].marshalled_input_size) * 8 / float(throughput_matrix[input_node_index][nodes[
            int(placement[0])].host]) + latency_matrix[input_node_index][nodes[int(placement[0])].host] / 1000.0

    total_latency = compute_latency + communication_latency + input_latency
    return total_latency


def get_throughput_matrix(nodes: List[controller_pb2.Node]) -> List[Dict[str, int]]:
    node_hosts = [node.host for node in nodes]
    throughput_matrix = []

    for node in nodes:
        throughputs = {
            network_throughput.host: network_throughput.throughput
            for network_throughput in node.state.resource_state.network_throughputs
        }

        # for those neighbours for which no measured throughput is known, use the node's configured bandwidth as placeholder
        for neighbour_host in node_hosts:
            if neighbour_host != node.host and neighbour_host not in throughputs:
                throughputs[neighbour_host] = node.state.resource_state.bandwidth

        throughput_matrix.append(throughputs)

    return throughput_matrix


def get_latency_matrix(nodes: List[controller_pb2.Node]) -> List[Dict[str, float]]:
    latency_matrix = []

    for node in nodes:
        latencys = {
            network_latency.host: network_latency.latency
            for network_latency in node.state.resource_state.network_latencies
        }

        latency_matrix.append(latencys)

    return latency_matrix
