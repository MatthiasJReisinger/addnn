import addnn.grpc
import addnn.logging
import asyncio
import click
import grpc
import grpc.aio
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import types
from addnn.cli import cli
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from addnn.node.proto import neural_network_pb2_grpc, node_pb2_grpc, node_state_pb2
from addnn.node.neural_network import NeuralNetwork
from addnn.node.neural_network_servicer import NeuralNetworkServicer
from addnn.node.node_servicer import NodeServicer
from addnn.node.node_state_monitor import NodeStateMonitor
from typing import Optional

iperf_server = None


def handle_sigterm(signum: int, frame: Optional[types.FrameType]) -> None:
    global iperf_server
    if iperf_server is not None:
        iperf_server.terminate()

    sys.exit(0)


@cli.command(name="node", help="Starts an ADDNN node.")
@click.option("--controller-host", required=True, help="The host that runs the controller.")
@click.option("--controller-port", type=int, required=True, help="The port at which to reach the controller.")
@click.option("--bind-ip",
              required=True,
              default="127.0.0.1",
              show_default=True,
              help="The IP to which the node should bind its API.")
@click.option("--bind-port",
              type=int,
              required=True,
              default=0,
              show_default=True,
              help="The port to which the node should bind its API.")
@click.option("--tier", type=int, required=True, default=0, show_default=True, help="The tier the node belongs to.")
@click.option("--memory",
              type=int,
              default=0,
              help="The maximal amount of RAM (bytes) that should be made available to the DNN.")
@click.option("--storage",
              type=int,
              default=0,
              help="The maximal amount of storage (bytes) that should be made available to the DNN.")
@click.option("--compute",
              type=int,
              default=0,
              help="The maximal amount of FLOPS that should be made available to the DNN.")
@click.option("--bandwidth", type=int, default=0, help="The node's available bandwidth.")
@click.option("--iperf", is_flag=True, default=False, help="Use iperf for throughput measurements.")
@click.option("--iperf-port",
              type=int,
              default=5201,
              show_default=True,
              help="On which port to run the node's iperf server.")
@click.option("--iperf-time",
              type=int,
              default=10,
              show_default=True,
              help="How long to run the iperf client each time network monitoring is triggered.")
@click.option("--is-input",
              is_flag=True,
              default=False,
              help="Determines whether the node acts as the input source for the DNN.")
@click.option("--model-cache-path",
              type=click.Path(),
              default="./model_cache",
              help="Path to a directory to cache the model at.")
@click.option("--use-existing-model-cache",
              is_flag=True,
              default=False,
              help="Whether to use the existing model cache.")
@click.option("--network-monitor-interval",
              type=int,
              default=10,
              show_default=True,
              help="Interval in seconds that determines how often to trigger network monitoring.")
def run(controller_host: str, controller_port: int, bind_ip: str, bind_port: int, tier: int, memory: int, storage: int,
        compute: int, bandwidth: int, iperf: bool, iperf_port: int, iperf_time: int, is_input: bool,
        model_cache_path: str, use_existing_model_cache: bool, network_monitor_interval: int) -> None:
    if is_input and tier != 0:
        raise Exception("only nodes on tier 0 can act as input nodes")

    signal.signal(signal.SIGTERM, handle_sigterm)

    # remove the model cache directory if desired
    if os.path.isdir(model_cache_path) and not use_existing_model_cache:
        shutil.rmtree(model_cache_path)

    # create the model cache directory if it does not exist yet
    if not os.path.isdir(model_cache_path):
        os.mkdir(model_cache_path)

    initial_resource_state = node_state_pb2.ResourceState()
    initial_resource_state.memory = memory
    initial_resource_state.storage = storage
    initial_resource_state.compute = compute
    initial_resource_state.bandwidth = bandwidth

    uds_address = "unix-abstract:addnn-node{}_{}.sock".format(bind_ip, bind_port)
    node_server_process = multiprocessing.Process(target=_run_node_server,
                                                  args=(bind_ip, bind_port, initial_resource_state, uds_address,
                                                        model_cache_path))
    node_server_process.daemon = True
    node_server_process.start()
    print("started node on port {}".format(bind_port))

    grpc_server_pid = node_server_process.pid
    if grpc_server_pid is None:
        raise Exception("cannot start gRPC server")

    _start_node_state_monitor(initial_resource_state, uds_address, model_cache_path, network_monitor_interval,
                              iperf_time, grpc_server_pid)

    if iperf:
        _start_iperf_server(iperf_port)

    node = controller_pb2.Node()
    node.host = bind_ip
    node.port = bind_port
    node.tier = tier
    node.is_input = is_input
    node.state.resource_state.CopyFrom(initial_resource_state)
    node.iperf_port = iperf_port

    _register_node_at_controller(controller_host, controller_port, node)

    node_server_process.join()


def _run_node_server(bind_ip: str, bind_port: int, initial_resource_state: node_state_pb2.ResourceState,
                     uds_address: str, model_cache_path: str) -> None:
    asyncio.run(_run_async_node_server(bind_ip, bind_port, initial_resource_state, uds_address, model_cache_path))


async def _run_async_node_server(bind_ip: str, bind_port: int, initial_resource_state: node_state_pb2.ResourceState,
                                 uds_address: str, model_cache_path: str) -> None:
    addnn.logging.init("node_server.log")

    neural_network = NeuralNetwork()

    # TODO since we now use a streaming gRPC to deploy layers, increasing the max message size might no longer be necessary
    server = grpc.aio.server(options=[('grpc.max_send_message_length',
                                       1024**3), ('grpc.max_receive_message_length', 1024**3)])

    node_servicer = NodeServicer(neural_network, initial_resource_state, model_cache_path)
    node_pb2_grpc.add_NodeServicer_to_server(node_servicer, server)

    neural_network_servicer = NeuralNetworkServicer(neural_network)
    neural_network_pb2_grpc.add_NeuralNetworkServicer_to_server(neural_network_servicer, server)

    server_endpoint = addnn.grpc.create_endpoint(bind_ip, bind_port)
    bind_port = server.add_insecure_port(address=server_endpoint)

    server.add_insecure_port(address=uds_address)

    await server.start()
    await server.wait_for_termination()


def _start_node_state_monitor(initial_resource_state: node_state_pb2.ResourceState, uds_address: str,
                              model_cache_path: str, network_monitor_interval: int, iperf_time: int,
                              grpc_server_pid: int) -> None:
    node_runtime_pid = grpc_server_pid
    node_state_monitor = NodeStateMonitor(initial_resource_state, network_monitor_interval, iperf_time, uds_address,
                                          model_cache_path, node_runtime_pid)
    node_state_monitor.daemon = True
    node_state_monitor.start()


def _start_iperf_server(iperf_port: int) -> None:
    global iperf_server
    iperf_server = subprocess.Popen(["iperf3", "-s", "-p", str(iperf_port)], stdout=subprocess.DEVNULL)


def _register_node_at_controller(controller_host: str, controller_port: int, node: controller_pb2.Node) -> None:
    controller_endpoint = addnn.grpc.create_endpoint(controller_host, controller_port)
    channel = grpc.insecure_channel(controller_endpoint)
    controller_stub = controller_pb2_grpc.ControllerStub(channel)
    register_node_request = controller_pb2.RegisterNodeRequest()
    register_node_request.node.CopyFrom(node)
    controller_stub.RegisterNode(register_node_request)
    print("registered node with controller at {}".format(controller_endpoint))
