from addnn.controller.proto.controller_pb2 import Node
from addnn.profile.layer_profile import LayerProfile
from addnn.serve.placement.placement import Placement, estimate_ent_to_end_latency
from addnn.serve.placement.strategy import Strategy
from collections import deque
from typing import List


class FirstFitDecreasing(Strategy):
    def name(self) -> str:
        return "ffd"

    def compute_placement(self, nodes: List[Node], layers: List[LayerProfile]) -> Placement:
        input_node_index = next(filter(lambda node_index: nodes[node_index].is_input, range(len(nodes))))
        sorted_node_indices = list(range(len(nodes)))
        sorted_node_indices.remove(input_node_index)
        sorted_node_indices.sort(key=lambda node_index: -nodes[node_index].state.resource_state.compute)
        remaining_node_indices = deque(sorted_node_indices)

        next_node_index = input_node_index
        available_memory = nodes[next_node_index].state.resource_state.memory
        available_storage = nodes[next_node_index].state.resource_state.storage

        placement = []

        for layer in layers:
            if available_memory < layer.in_memory_size or available_storage < layer.storage_size:
                next_node_index = remaining_node_indices.popleft()
                available_memory = nodes[next_node_index].state.resource_state.memory
                available_storage = nodes[next_node_index].state.resource_state.storage

            available_memory -= layer.in_memory_size
            available_storage -= layer.storage_size

            placement.append(next_node_index)

        return placement
