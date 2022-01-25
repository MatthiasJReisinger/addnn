import addnn
import addnn.cli
import addnn.model
import addnn.serve.scheduler
import click
import grpc
import pickle
import pulp
import torch
import time
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from addnn.node.proto import node_pb2, node_pb2_grpc
from addnn.util.torchscript import read_model_from_preprocessed_script
from addnn.util.serialization import serialize_module
from addnn.profile.layer_profile import LayerProfile, get_layer_profiles
from addnn.serve.placement import strategy_loader
from addnn.serve.placement.placement import Placement
from addnn.serve.placement.strategy import Strategy
from google.protobuf.empty_pb2 import Empty
from typing import Dict, List, Optional


@addnn.cli.cli.command("serve", help="Serves an ADDNN model over a distributed compute hierarchy.")
@click.option("--controller-host", required=True, help="The host that runs the controller.")
@click.option("--controller-port", required=True, type=int, help="The port at which to reach the controller.")
@click.option("--placement",
              "placement_type",
              required=True,
              type=click.Choice(strategy_loader.get_available_strategy_names()),
              help="How to place layers on nodes.")
@click.option("--repeat-interval", type=int, help="Repeatedly serve the model after the given time interval (seconds).")
@click.option("--torchscript", is_flag=True, default=False)
@click.option("--input-shape", help="shape of the given model (e.g., '3,224,224').")
@click.option("--profile", "profile_path", type=click.Path(exists=True), help="Path to a model profile.")
@click.option("--json-report-dir",
              "json_report_path",
              default="./scheduler-reports",
              type=click.Path(),
              help="Export a report in json format after each scheduler iteration in a directory at the given path.")
@click.option(
    "--rpc-overhead-factor",
    type=float,
    default=1.0,
    show_default=True,
    help=
    "Under certain conditions, monitored network throughput/latency levels might not be sufficient to reflect the actual RPC throughput/latency between nodes. Therefore, this constant factor can be used to linearly scale the schdeduler's communication overhead predictions, in order to avoid sub-optimal placement decision."
)
@click.argument("filename", type=click.Path(exists=True))
def cli(controller_host: str, controller_port: int, placement_type: str, repeat_interval: Optional[int],
        torchscript: bool, input_shape: str, profile_path: str, json_report_path: str, rpc_overhead_factor: float,
        filename: str) -> None:
    addnn.serve.scheduler.run(controller_host, controller_port, placement_type, repeat_interval, torchscript,
                              input_shape, profile_path, json_report_path, rpc_overhead_factor, filename)
