import addnn
import addnn.model
import gc
import icmplib
import logging
import os
import psutil
import shutil
import time
import torch
from addnn.node.proto import node_pb2, node_pb2_grpc, node_state_pb2
from addnn.node.neural_network import NeuralNetwork, Model
from addnn.controller.proto import controller_pb2
from addnn.util.serialization import deserialize_module
from google.protobuf.empty_pb2 import Empty
from typing import Iterable, List

logger = logging.getLogger(__name__)


class NodeServicer(node_pb2_grpc.NodeServicer):
    """
    Implementation of the Node service interface.
    """
    def __init__(self, neural_network: NeuralNetwork, initial_resource_state: node_state_pb2.ResourceState,
                 model_cache_path: str):
        super().__init__()
        self._neural_network = neural_network
        self._resource_state = initial_resource_state
        self._neighbour_nodes: Iterable[controller_pb2.Node] = []
        self._model_cache_path = model_cache_path
        self._number_of_cached_layers = len(os.listdir(model_cache_path))

    async def DeployModel(self, request_iterator, context):  # type: ignore
        logger.debug("receiving model")
        start = time.time()

        layer_index = 0
        async for request in request_iterator:
            logger.debug("receiving layer {}".format(layer_index))
            _save_layer(request, layer_index, self._model_cache_path)
            layer_index += 1

        self._number_of_cached_layers = layer_index

        end = time.time()
        duration = end - start
        logger.debug("received model in {} seconds".format(duration))
        return Empty()

    async def DeleteModel(self, request, context):  # type: ignore
        if os.path.isdir(self._model_cache_path):
            shutil.rmtree(self._model_cache_path)
        os.mkdir(self._model_cache_path)
        self._number_of_cached_layers = 0
        return Empty()

    async def ActivateLayers(self, request, context):  # type: ignore
        # unload all currently active layers
        self._neural_network.model = None
        gc.collect()

        # TODO set new exit rates when loading layers from disk (has to be retrieved from scheduler)

        # load new active layers
        active_layers = []
        if request.HasField("active_layers"):
            logger.debug("activate layers {} to {}".format(request.active_layers.start_index,
                                                           request.active_layers.end_index))
            for layer_index in range(request.active_layers.start_index, request.active_layers.end_index + 1):
                marshalled_layer = _load_layer(self._model_cache_path, layer_index)
                layer = _unmarshal_layer(marshalled_layer)
                active_layers.append(layer)

        remote_layer = None
        if request.HasField("remote_layer"):
            remote_layer = request.remote_layer
            logger.debug("activate proxy layer for next layer at {}".format(
                addnn.grpc.create_endpoint(remote_layer.host, remote_layer.port)))

        neural_network_model = None
        if len(active_layers) > 0 or remote_layer is not None:
            neural_network_model = Model(active_layers, remote_layer, request.active_layers.start_index)

        self._neural_network.model = neural_network_model

        return Empty()

    async def DeactivateLayers(self, request, context):  # type: ignore
        self._neural_network.model = None
        gc.collect()
        return Empty()

    async def ReadNodeState(self, request, context):  # type: ignore
        self._neighbour_nodes = list(request.neighbour_nodes)
        response = node_pb2.ReadNodeStateResponse()

        # append the node's resource state to the response
        response.node_state.resource_state.CopyFrom(self._resource_state)

        for layer_index in range(self._number_of_cached_layers):
            layer_state = node_state_pb2.LayerState()
            layer_state.layer_index = layer_index
            layer_state.active = False
            response.node_state.neural_network_state.layer_states.append(layer_state)

        # if the node currently hosts layers, then append the states of the layers to the response
        if self._neural_network.model is not None:
            layer_index = self._neural_network.model.start_layer_index
            for layer in self._neural_network.model.layers:
                layer_state = response.node_state.neural_network_state.layer_states[layer_index]
                layer_state.active = True

                if layer.exit_branch is not None:
                    layer_state.number_of_exited_samples = layer.exit_branch.number_of_exited_samples

                layer_index += 1

        return response

    async def UpdateResourceState(self, request, context):  # type: ignore
        self._resource_state = request.resource_state
        return Empty()

    async def ReadNeighbourNodes(self, request, context):  # type: ignore
        response = node_pb2.ReadNeighbourNodesResponse()
        response.neighbour_nodes.extend(self._neighbour_nodes)
        return response


def _unmarshal_layer(marshalled_layer: node_pb2.LocalLayer) -> addnn.model.Layer:
    main_branch = None
    if len(marshalled_layer.main_branch) > 0:
        main_branch = deserialize_module(marshalled_layer.main_branch, marshalled_layer.is_torchscript)

    exit_branch = None
    if marshalled_layer.HasField("exit_branch"):
        exit_classifier = deserialize_module(marshalled_layer.exit_branch.classifier, marshalled_layer.is_torchscript)
        confidence_threshold = marshalled_layer.exit_branch.confidence_threshold
        exit_branch = addnn.model.Exit(exit_classifier, confidence_threshold)

    unmarshalled_layer = addnn.model.Layer(main_branch, exit_branch)
    return unmarshalled_layer


def _save_layer(layer: node_pb2.LocalLayer, layer_index: int, model_cache_path: str) -> None:
    path = "{}/layer{}.pb".format(model_cache_path, layer_index)
    with open(path, "wb") as layer_file:
        layer_file.write(layer.SerializeToString())


def _load_layer(model_cache_path: str, layer_index: int) -> node_pb2.LocalLayer:
    path = "{}/layer{}.pb".format(model_cache_path, layer_index)
    with open(path, "rb") as layer_file:
        layer = node_pb2.LocalLayer()
        layer.ParseFromString(layer_file.read())
        return layer
