from addnn.controller.proto.controller_pb2 import Node
from addnn.profile.layer_profile import LayerProfile
from addnn.serve.placement.placement import NodeIndex, Placement
from addnn.serve.placement.strategy import Strategy
from typing import Iterable, List


class CloudStrategy(Strategy):
    """
    Strategy that places all layers on a single node in the cloud tier with the minimal end-to-end latency.
    """
    def name(self) -> str:
        return "cloud"

    def compute_placement(self, nodes: List[Node], layers: List[LayerProfile]) -> Placement:
        candidate_node_index = get_cloud_node_with_minimal_inference_latency(nodes, layers)
        placement = [candidate_node_index] * len(layers)
        return placement


def get_cloud_node_with_minimal_inference_latency(nodes: List[Node], layers: List[LayerProfile]) -> NodeIndex:
    cloud_tier = max([node.tier for node in nodes])
    input_node = next(filter(lambda node: node.is_input, nodes))

    total_compute_demand = sum([layer.execution_probability * layer.flops for layer in layers])
    input_node_throughput = {
        network_throughput.host: network_throughput.throughput
        for network_throughput in input_node.state.resource_state.network_throughputs
    }
    input_node_latency = {
        network_latency.host: network_latency.latency
        for network_latency in input_node.state.resource_state.network_latencies
    }

    candidate_node_index = 0
    min_latency = None
    for node_index in range(len(nodes)):
        if nodes[node_index].tier == cloud_tier:
            compute_latency = total_compute_demand / float(nodes[node_index].state.resource_state.compute)
            input_latency = layers[0].marshalled_input_size * 8 / input_node_throughput[
                nodes[node_index].host] + input_node_latency[nodes[node_index].host] / 1000.0
            latency = input_latency + compute_latency
            if min_latency is None or latency < min_latency:
                min_latency = latency
                candidate_node_index = node_index

    return candidate_node_index
