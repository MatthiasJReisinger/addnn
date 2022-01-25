import addnn.cli
import addnn.grpc
import click
import grpc
import ipaddress
import signal
import sys
import time
import types
from addnn.controller.controller_servicer import ControllerServicer
from addnn.controller.node_state_monitor import NodeStateMonitor
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from addnn.node.proto import node_pb2, node_pb2_grpc
from concurrent.futures import ThreadPoolExecutor
from google.protobuf.empty_pb2 import Empty
from typing import Optional


def handle_sigterm(signum: int, frame: Optional[types.FrameType]) -> None:
    sys.exit(0)


@addnn.cli.cli.command("controller", help="Starts an ADDNN controller.")
@click.option("--bind-ip", default="127.0.0.1", show_default=True, help="The IP at which to expose the controller API.")
@click.option("--bind-port",
              type=int,
              default=0,
              help="The IP at which to expose the controller API [default: a random port].")
@click.option("--node-monitor-interval",
              type=int,
              default=2,
              show_default=True,
              help="The interval (seconds) at which the state of nodes should be reloaded.")
def cli(bind_ip: str, bind_port: int, node_monitor_interval: int) -> None:
    run(bind_ip, bind_port, node_monitor_interval)


def run(bind_ip: str, bind_port: int, node_monitor_interval: int) -> None:
    signal.signal(signal.SIGTERM, handle_sigterm)

    server = grpc.server(ThreadPoolExecutor(max_workers=10))

    controller = ControllerServicer()
    controller_pb2_grpc.add_ControllerServicer_to_server(controller, server)

    server_endpoint = addnn.grpc.create_endpoint(bind_ip, bind_port)
    port = server.add_insecure_port(address=server_endpoint)

    uds_address = "unix-abstract:addnn-controller{}_{}.sock".format(bind_ip, bind_port)
    server.add_insecure_port(address=uds_address)

    node_state_monitor = NodeStateMonitor(node_monitor_interval, uds_address)
    node_state_monitor.daemon = True
    node_state_monitor.start()

    server.start()
    print("started controller on port {}".format(port))
    server.wait_for_termination()
