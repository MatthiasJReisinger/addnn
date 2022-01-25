import addnn
import addnn.cli
import addnn.grpc
import addnn.model
import click
import grpc
import os
import pickle
import shutil
import torch
import time
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from addnn.node.proto import node_pb2, node_pb2_grpc
from addnn.util.torchscript import read_model_from_preprocessed_script
from addnn.util.serialization import serialize_module
from addnn.profile.layer_profile import LayerProfile, get_layer_profiles
from addnn.serve.placement import strategy_loader
from addnn.serve.placement.placement import NodeIndex, Placement, estimate_ent_to_end_latency
from addnn.serve.placement.strategy import Strategy
from addnn.serve.proto import scheduler_pb2
from google.protobuf.empty_pb2 import Empty
from google.protobuf.json_format import MessageToJson
from typing import Dict, Generator, Iterable, List, Optional


def run(controller_host: str, controller_port: int, placement_type: str, repeat_interval: Optional[int],
        torchscript: bool, input_shape: str, profile_path: Optional[str], json_report_path: str,
        rpc_overhead_factor: float, model_path: str) -> None:
    if torchscript:
        model = read_model_from_preprocessed_script(model_path, input_shape)
    else:
        model = torch.load(model_path)
        model.eval()

    layers = model.layers

    if profile_path is not None:
        with open(profile_path, "rb") as profile_file:
            layer_profiles = pickle.load(profile_file)
    else:
        layer_profiles = get_layer_profiles(layers)

    # initialize the directory for storing the scheduler reports
    if os.path.isdir(json_report_path):
        shutil.rmtree(json_report_path)
    os.mkdir(json_report_path)

    scheduler_run_index = 0

    if repeat_interval is not None:
        if repeat_interval <= 0:
            raise Exception("repeat interval has to be > 0")

        while True:
            serve(controller_host, controller_port, placement_type, layers, layer_profiles, scheduler_run_index,
                  json_report_path, rpc_overhead_factor)
            time.sleep(repeat_interval)
            scheduler_run_index += 1
    else:
        serve(controller_host, controller_port, placement_type, layers, layer_profiles, scheduler_run_index,
              json_report_path, rpc_overhead_factor)


def _has_model(node: controller_pb2.Node) -> bool:
    return len(node.state.neural_network_state.layer_states) > 0


def serve(controller_host: str, controller_port: int, placement_type: str, layers: List[addnn.model.Layer],
          layer_profiles: List[addnn.profile.layer_profile.LayerProfile], scheduler_run_index: int,
          json_report_path: str, rpc_overhead_factor: float) -> None:
    report = scheduler_pb2.Report()
    report.start_timestamp = int(time.time_ns() / 1000)

    controller_endpoint = addnn.grpc.create_endpoint(controller_host, controller_port)
    channel = grpc.insecure_channel(controller_endpoint)
    controller_stub = controller_pb2_grpc.ControllerStub(channel)

    list_nodes_response = controller_stub.ListNodes(Empty())
    nodes = [registered_node.node for registered_node in list_nodes_response.nodes]

    # deploy layers to nodes that don't have them yet
    for node in nodes:
        if not _has_model(node):
            _deploy_layers_to_node(node, layers)

    # adjust monitored network conditions based given RPC overhead factor
    nodes_with_adjusted_network_conditions = _adjust_monitored_network_conditions(nodes, rpc_overhead_factor)

    current_placement = get_current_placement(nodes_with_adjusted_network_conditions, len(layer_profiles))

    # validate that there exists exactly one input node in the compute hierarchy
    node_indices = range(len(nodes))
    input_node_indices = list(filter(lambda node_index: nodes[node_index].is_input, node_indices))
    number_of_input_nodes = len(input_node_indices)
    if number_of_input_nodes != 1:
        raise Exception(
            "there has to be exactly one input node, but currenly there are {}".format(number_of_input_nodes))
    input_node_index = input_node_indices[0]

    new_placement = _compute_placement(placement_type, nodes, layer_profiles)
    estimated_latency = estimate_ent_to_end_latency(new_placement, nodes_with_adjusted_network_conditions,
                                                    layer_profiles)
    print("predicted end-to-end latency: {}".format(estimated_latency))
    print("predicted inference throughput: {}".format(1 / estimated_latency))

    if new_placement != current_placement:
        _activate_layers_on_nodes(nodes, input_node_index, new_placement)
    else:
        print("placement did not change")

    # export a report about the current scheduler run as json file
    report.end_timestamp = int(time.time_ns() / 1000)
    report.nodes.extend(list_nodes_response.nodes)
    report.placement.nodes.extend([list_nodes_response.nodes[node_index].uuid for node_index in new_placement])
    report.estimated_latency = estimated_latency
    for layer_profile in layer_profiles:
        report.layer_profiles.append(_parse_layer_profile(layer_profile))
    json_file_name = "{}/schedule{}.json".format(json_report_path, scheduler_run_index)
    with open(json_file_name, "w") as json_file:
        json_file.write(MessageToJson(report))


def _parse_layer_profile(layer_profile: LayerProfile) -> scheduler_pb2.LayerProfile:
    pb_layer_profile = scheduler_pb2.LayerProfile()
    pb_layer_profile.flops = layer_profile.flops
    pb_layer_profile.in_memory_size = layer_profile.in_memory_size
    pb_layer_profile.storage_size = layer_profile.storage_size
    pb_layer_profile.marshalled_input_size = layer_profile.marshalled_input_size
    pb_layer_profile.has_exit = layer_profile.has_exit
    pb_layer_profile.number_of_exited_samples = layer_profile.number_of_exited_samples
    pb_layer_profile.exit_probability = layer_profile.exit_probability
    pb_layer_profile.execution_probability = layer_profile.execution_probability
    for operator in layer_profile.operators:
        pb_layer_profile.operators.append(operator)
    return pb_layer_profile


def _adjust_monitored_network_conditions(nodes: List[controller_pb2.Node],
                                         rpc_overhead_factor: float) -> List[controller_pb2.Node]:
    adjusted_nodes = []
    for node in nodes:
        adjusted_node = controller_pb2.Node()
        adjusted_node.CopyFrom(node)
        adjusted_nodes.append(adjusted_node)

        for network_throughput in adjusted_node.state.resource_state.network_throughputs:
            network_throughput.throughput = int(network_throughput.throughput / rpc_overhead_factor)

        for network_latency in adjusted_node.state.resource_state.network_latencies:
            network_latency.latency = network_latency.latency * rpc_overhead_factor

    return adjusted_nodes


LayerIndex = int


# TODO don't use integer indices to represent placement when comparing to old placement
def get_current_placement(nodes: List[controller_pb2.Node], number_of_layers: int) -> Optional[Placement]:
    placement_dict: Dict[LayerIndex, NodeIndex] = dict()
    for node_index, node in enumerate(nodes):
        for layer_state in node.state.neural_network_state.layer_states:
            if layer_state.active:
                placement_dict[layer_state.layer_index] = node_index

    if len(placement_dict) == number_of_layers:
        placement = [placement_dict[layer_index] for layer_index in range(number_of_layers)]
        return placement
    else:
        return None


def _compute_placement(placement_type: str, nodes: List[controller_pb2.Node],
                       layer_profiles: List[addnn.profile.layer_profile.LayerProfile]) -> Placement:
    strategy = strategy_loader.load_strategy(placement_type)
    return strategy.compute_placement(nodes, layer_profiles)


def _activate_layers_on_nodes(nodes: List[controller_pb2.Node], input_node_index: int, placement: Placement) -> None:
    activation_start = time.time()

    # for each node, create a list of layers that should be deployed to the node
    layers_per_node: List[List[int]] = [[] for node_index in range(len(nodes))]
    for layer_index in range(len(placement)):
        node_index = placement[layer_index]
        layers_per_node[node_index].append(layer_index)

    # actually activate the layers on the nodes
    for node_index in range(len(nodes)):
        # get the layers that should be deployed to the current node
        node_layer_indices = layers_per_node[node_index]
        node = nodes[node_index]

        if len(node_layer_indices) == 0:
            if _has_active_layers(node):
                print("deactivate layers on node at {}".format(addnn.grpc.create_endpoint(node.host, node.port)))
                _deactivate_layers_on_node(node)
            else:
                print("node {} has no active layers".format(addnn.grpc.create_endpoint(node.host, node.port)))
        else:
            _activate_layers_on_node(node, node_layer_indices, placement, nodes)

    # if the input node does not host any layers, then let it redirect the DNN input to the node that hosts the first layer
    if len(layers_per_node[input_node_index]) == 0:
        print("activate proxy layer on input node")
        input_node = nodes[input_node_index]
        node_with_first_layer = nodes[placement[0]]
        _activate_proxy_layer_on_node(input_node, node_with_first_layer)

    activation_end = time.time()
    activation_duration = activation_end - activation_start
    print("activated layers on nodes in {} seconds".format(activation_duration))


def _deploy_layers_to_node(node: controller_pb2.Node, layers: List[addnn.model.Layer]) -> None:
    node_endpoint = addnn.grpc.create_endpoint(node.host, node.port)
    with grpc.insecure_channel(node_endpoint) as channel:
        node_stub = node_pb2_grpc.NodeStub(channel)
        node_stub.DeployModel(_generate_layers(node, layers))


def _generate_layers(node: controller_pb2.Node,
                     layers: List[addnn.model.Layer]) -> Generator[node_pb2.LocalLayer, None, None]:
    for layer in layers:
        pb_layer = node_pb2.LocalLayer()

        if layer.main_branch is not None:
            pb_layer.is_torchscript = isinstance(layer.main_branch, torch.jit.ScriptModule)
            pb_layer.main_branch = serialize_module(layer.main_branch)

        if layer.exit_branch is not None:
            pb_layer.is_torchscript = isinstance(layer.exit_branch.classifier, torch.jit.ScriptModule)
            pb_layer.exit_branch.classifier = serialize_module(layer.exit_branch.classifier)
            pb_layer.exit_branch.confidence_threshold = layer.exit_branch.confidence_threshold

        yield pb_layer


def _has_active_layers(node: controller_pb2.Node) -> bool:
    for layer_state in node.state.neural_network_state.layer_states:
        if layer_state.active:
            return True

    return False


def _activate_layers_on_node(node: controller_pb2.Node, layers: List[LayerIndex], placement: Placement,
                             nodes: List[controller_pb2.Node]) -> None:
    activate_layers_request = node_pb2.ActivateLayersRequest()

    node_endpoint = addnn.grpc.create_endpoint(node.host, node.port)

    active_layers = node_pb2.LayerRange()
    active_layers.start_index = min(layers)
    active_layers.end_index = max(layers)
    activate_layers_request.active_layers.CopyFrom(active_layers)
    print("activate layers {} to {} on node at {}".format(active_layers.start_index, active_layers.end_index,
                                                          node_endpoint))

    # if this node does not host all remaining layers, then determine the node that hosts the next layer
    remote_layer = None
    max_layer_index = len(placement) - 1
    if active_layers.end_index < max_layer_index:
        next_layer_index = active_layers.end_index + 1
        next_node_index = placement[next_layer_index]
        next_node = nodes[next_node_index]
        activate_layers_request.remote_layer.host = next_node.host
        activate_layers_request.remote_layer.port = next_node.port

    with addnn.grpc.create_channel(node.host, node.port) as channel:
        node_stub = node_pb2_grpc.NodeStub(channel)
        node_stub.ActivateLayers(activate_layers_request)


def _activate_proxy_layer_on_node(node: controller_pb2.Node, next_node: controller_pb2.Node) -> None:
    activate_layers_request = node_pb2.ActivateLayersRequest()
    activate_layers_request.remote_layer.host = next_node.host
    activate_layers_request.remote_layer.port = next_node.port

    with addnn.grpc.create_channel(node.host, node.port) as channel:
        node_stub = node_pb2_grpc.NodeStub(channel)
        node_stub.ActivateLayers(activate_layers_request)


def _deactivate_layers_on_node(node: controller_pb2.Node) -> None:
    with addnn.grpc.create_channel(node.host, node.port) as channel:
        node_stub = node_pb2_grpc.NodeStub(channel)
        node_stub.DeactivateLayers(Empty())
