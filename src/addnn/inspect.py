import addnn
import addnn.grpc
import click
import csv
import grpc
import pickle
import torch
from addnn.cli import cli
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from addnn.util.torchscript import read_model_from_preprocessed_script
from addnn.profile.layer_profile import LayerProfile, get_layer_profiles
from google.protobuf.empty_pb2 import Empty
from pathlib import Path


@cli.group()
def inspect() -> None:
    pass


@inspect.command("nodes", help="Inspect the compute nodes that are currently registered at the controller.")
@click.option("--controller-host", required=True, help="The host that runs the controller.")
@click.option("--controller-port", required=True, type=int, help="The port at which to reach the controller.")
def nodes(controller_host: str, controller_port: int) -> None:
    controller_stub = addnn.grpc.create_tcp_stub(controller_pb2_grpc.ControllerStub, controller_host, controller_port)
    list_nodes_response = controller_stub.ListNodes(Empty())
    nodes = list_nodes_response.nodes
    print("{}".format(nodes))
