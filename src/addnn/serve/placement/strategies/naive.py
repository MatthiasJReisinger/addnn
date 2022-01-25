import numpy
from addnn.serve.placement.strategy import Strategy
from addnn.controller.proto.controller_pb2 import Node
from addnn.profile.layer_profile import LayerProfile
from addnn.serve.placement.placement import NodeIndex, Placement, estimate_ent_to_end_latency, get_throughput_matrix, get_latency_matrix
from addnn.serve.placement.strategy import Strategy
from typing import List


class NaiveStrategy(Strategy):
    """
    Distribute the layers evenly between all nodes.
    """
    def name(self) -> str:
        return "naive"

    def compute_placement(self, nodes: List[Node], layers: List[LayerProfile]) -> Placement:
        layer_indices = numpy.arange(len(layers))
        layers_per_node = numpy.array_split(layer_indices, len(nodes))
        node_indices = range(len(nodes))
        placement = [node_index for node_index in node_indices for layer_index in layers_per_node[node_index]]
        return placement
