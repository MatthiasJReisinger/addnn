import addnn
import addnn.serve.scheduler
import grpc
import multiprocessing
import time
from addnn.benchmark.end_to_end.config import BenchmarkConfig, NodeConfig, SchedulerConfig, read_node_configs
from addnn.benchmark.end_to_end.remote_control.host import *
from addnn.controller.proto.controller_pb2_grpc import ControllerStub
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from google.protobuf.empty_pb2 import Empty
from typing import Dict, Iterable, List, Optional


def reset_remote_controller(host: str, user: str, ssh_key_path: Optional[str]) -> None:
    print("reset controller at {}".format(host))

    command = "cat ~/controller.pid 2> /dev/null | xargs kill &> /dev/null ; rm -f ~/controller.pid"
    execute_remote_command(host, user, command, ssh_key_path)


def start_remote_controller(host: str, user: str, bind_port: int, ssh_key_path: Optional[str]) -> None:
    print("start controller at {}".format(host))

    addnn_executable = "/home/ubuntu/.poetry/bin/poetry run addnn"  # TODO

    command = "cd addnn ; "
    command += "nohup {} controller ".format(addnn_executable)
    command += "--bind-ip={} ".format(host)
    command += "--bind-port={} ".format(bind_port)
    command += "&> /home/ubuntu/controller-log.txt & echo $! > ~/controller.pid"
    execute_remote_command(host, user, command, ssh_key_path)


def start_controller(benchmark_config: BenchmarkConfig) -> None:
    if benchmark_config.start_controller:
        if benchmark_config.controller_user is None:
            raise Exception("please provide a value for --controller-user")
        reset_remote_controller(benchmark_config.controller_host, benchmark_config.controller_user,
                                benchmark_config.controller_ssh_key)
        start_remote_controller(benchmark_config.controller_host, benchmark_config.controller_user,
                                benchmark_config.controller_port, benchmark_config.controller_ssh_key)
        wait_process = multiprocessing.Process(target=_wait_until_controller_started,
                                               args=(benchmark_config.controller_host,
                                                     benchmark_config.controller_port))
        wait_process.start()
        wait_process.join()


def _wait_until_controller_started(controller_host: str, controller_port: int) -> None:
    print("wait for controller to start")

    while not _is_controller_reachable(controller_host, controller_port):
        time.sleep(1)


def _is_controller_reachable(controller_host: str, controller_port: int) -> bool:
    try:
        controller_stub = addnn.grpc.create_tcp_stub(ControllerStub, controller_host, controller_port)
        list_nodes_response = controller_stub.ListNodes(Empty())
        return True
    except Exception as e:
        return False


def get_current_placement(controller_host: str, controller_port: int,
                          number_of_layers: int) -> Optional[List[controller_pb2.RegisteredNode]]:
    controller_endpoint = addnn.grpc.create_endpoint(controller_host, controller_port)
    channel = grpc.insecure_channel(controller_endpoint)
    controller_stub = controller_pb2_grpc.ControllerStub(channel)
    list_nodes_response = controller_stub.ListNodes(Empty())
    nodes = list_nodes_response.nodes

    placement_dict: Dict[int, controller_pb2.RegisteredNode] = dict()
    for node in nodes:
        for layer_state in node.node.state.neural_network_state.layer_states:
            if layer_state.active:
                placement_dict[layer_state.layer_index] = node

    if len(placement_dict) == number_of_layers:
        placement = [placement_dict[layer_index] for layer_index in range(number_of_layers)]
        return placement
    else:
        return None


def is_initial_deployment_finished(controller_host: str, controller_port: int, number_of_layers: int) -> bool:
    controller_stub = addnn.grpc.create_tcp_stub(ControllerStub, controller_host, controller_port)
    list_nodes_response = controller_stub.ListNodes(Empty())
    nodes = [registered_node.node for registered_node in list_nodes_response.nodes]
    return addnn.serve.scheduler.get_current_placement(nodes, number_of_layers) is not None
