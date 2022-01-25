import addnn.grpc
import click
import grpc
import multiprocessing
import time
from addnn.cli import cli
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from addnn.node.proto import node_pb2, node_pb2_grpc, node_state_pb2
from google.protobuf.empty_pb2 import Empty
from typing import Iterable


class NodeStateMonitor(multiprocessing.Process):
    """
    Monitors the state of the compute nodes that are registered at the controller.
    """
    def __init__(self, interval: int, uds_address: str) -> None:
        super().__init__()
        self._interval = interval
        self._uds_address = uds_address

    def run(self) -> None:
        while True:
            self._reload_node_states()
            time.sleep(self._interval)

    def _reload_node_states(self) -> None:
        controller_channel = grpc.insecure_channel(self._uds_address)
        controller_stub = controller_pb2_grpc.ControllerStub(controller_channel)
        list_nodes_response = controller_stub.ListNodes(Empty())

        for registered_node in list_nodes_response.nodes:
            _reload_node_state(registered_node, list_nodes_response.nodes, controller_stub)


def _reload_node_state(registered_node: controller_pb2.RegisteredNode, nodes: Iterable[controller_pb2.RegisteredNode],
                       controller_stub: controller_pb2_grpc.ControllerStub) -> None:
    try:
        node_id = registered_node.uuid
        node = registered_node.node
        node_state = _read_node_state(node_id, node, nodes)
        _update_node_state(node_id, node_state, controller_stub)
    except grpc.RpcError as e:
        node_endpoint = addnn.grpc.create_endpoint(node.host, node.port)
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            print("could not reach node {} at {}, deregister it".format(node_id, node_endpoint))
        else:
            print("could not obtain state of {} node at {} due to error:\n{}".format(node_id, node_endpoint, str(e)))

        _deregister_node(registered_node.uuid, controller_stub)


def _read_node_state(node_id: str, node: controller_pb2.Node,
                     nodes: Iterable[controller_pb2.RegisteredNode]) -> node_state_pb2.NodeState:
    node_stub = addnn.grpc.create_tcp_stub(node_pb2_grpc.NodeStub, node.host, node.port)
    request = node_pb2.ReadNodeStateRequest()

    for registered_neighbour_node in nodes:
        if registered_neighbour_node.uuid != node_id:
            request.neighbour_nodes.append(registered_neighbour_node.node)

    response = node_stub.ReadNodeState(request)
    return response.node_state


def _update_node_state(node_id: str, node_state: node_state_pb2.NodeState,
                       controller_stub: controller_pb2_grpc.ControllerStub) -> None:
    try:
        request = controller_pb2.UpdateNodeStateRequest()
        request.uuid = node_id
        request.node_state.CopyFrom(node_state)
        controller_stub.UpdateNodeState(request)
    except grpc.RpcError as e:
        print("could not update node state of node {} due to error:\n{}".format(node_id, str(e)))


def _deregister_node(node_id: str, controller_stub: controller_pb2_grpc.ControllerStub) -> None:
    try:
        request = controller_pb2.DeregisterNodeRequest()
        request.uuid = node_id
        controller_stub.DeregisterNode(request)
    except grpc.RpcError as e:
        print("could not deregister node {} due to error:\n{}".format(node_id, str(e)))
