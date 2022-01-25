import addnn.grpc
import click
import grpc
import threading
import time
import uuid
from addnn.cli import cli
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from addnn.node.proto import node_pb2, node_pb2_grpc
from concurrent.futures import ThreadPoolExecutor
from google.protobuf.empty_pb2 import Empty
from typing import Dict


class ControllerServicer(controller_pb2_grpc.ControllerServicer):
    """
    Implementation of the Controller service interface.
    """
    def __init__(self) -> None:
        super().__init__()
        self._nodes: Dict[str, controller_pb2.Node] = {}

    def RegisterNode(self, request, context):  # type: ignore
        node_id = str(uuid.uuid4())
        self._nodes[node_id] = request.node
        node_endpoint = addnn.grpc.create_endpoint(request.node.host, request.node.port)
        print("register node at {}".format(node_endpoint))
        response = controller_pb2.RegisterNodeResponse()
        response.uuid = node_id
        return response

    def DeregisterNode(self, request, context):  # type: ignore
        if not request.uuid in self._nodes:
            print("{} does not refer to a registered node")

        self._nodes.pop(request.uuid)
        return Empty()

    def UpdateNodeState(self, request, context):  # type: ignore
        if not request.uuid in self._nodes:
            print("{} does not refer to a registered node")

        self._nodes[request.uuid].state.CopyFrom(request.node_state)
        return Empty()

    def ListNodes(self, request, context):  # type: ignore
        response = controller_pb2.ListNodesResponse()

        for node_id, node in self._nodes.items():
            registered_node = controller_pb2.RegisteredNode()
            registered_node.uuid = node_id
            registered_node.node.CopyFrom(node)
            response.nodes.append(registered_node)

        return response
