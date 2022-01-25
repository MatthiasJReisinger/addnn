from addnn.controller.proto.controller_pb2 import Node
from addnn.profile.layer_profile import LayerProfile
from addnn.serve.placement.placement import NodeIndex, Placement
from addnn.serve.placement.strategy import Strategy
from addnn.serve.placement.strategies.cloud import get_cloud_node_with_minimal_inference_latency
from typing import Iterable, List


class DeviceExitStrategy(Strategy):
    """
    Strategy that places all layers up to the first side-exit on the input device node.
    """
    def name(self) -> str:
        return "device-exit"

    def compute_placement(self, nodes: List[Node], layers: List[LayerProfile]) -> Placement:
        # determine index of layer with first side exit
        first_side_exit_layer_index = next(filter(lambda layer_index: layers[layer_index].has_exit, range(len(layers))))
        cloud_layers = layers[first_side_exit_layer_index + 1:]
        cloud_node_index = get_cloud_node_with_minimal_inference_latency(nodes, cloud_layers)

        # place all layers up to side-exit on device node, place all remaining layers on cloud node
        input_node_index = next(filter(lambda node_index: nodes[node_index].is_input, range(len(nodes))))
        number_of_device_layers = first_side_exit_layer_index + 1
        device_placement = [input_node_index] * number_of_device_layers
        number_of_cloud_layers = len(layers) - number_of_device_layers
        cloud_placement = [cloud_node_index] * number_of_cloud_layers
        placement = device_placement + cloud_placement
        return placement
