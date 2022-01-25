import addnn
import addnn.serve.scheduler
import grpc
import multiprocessing
import subprocess
import time
from addnn.benchmark.end_to_end.config import BenchmarkConfig, NodeConfig, SchedulerConfig, read_node_configs
from addnn.benchmark.end_to_end.remote_control.host import *
from addnn.controller.proto.controller_pb2_grpc import ControllerStub
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from google.protobuf.empty_pb2 import Empty
from typing import Dict, Iterable, List, Optional


def reset_scheduler(scheduler_config: SchedulerConfig) -> None:
    print("reset scheduler at {}".format(scheduler_config.host))

    command = "cat ~/scheduler.pid 2> /dev/null | xargs kill &> /dev/null ; rm -f ~/scheduler.pid"
    execute_remote_command(scheduler_config.host, scheduler_config.user, command, scheduler_config.ssh_key_path)


def run_scheduler_once(scheduler_config: SchedulerConfig, controller_host: str, controller_port: int,
                       placement_type: str, input_shape: Iterable[int]) -> None:
    print("start scheduler at {}".format(scheduler_config.host))

    addnn_executable = "/home/ubuntu/.poetry/bin/poetry run addnn"  # TODO

    command = "cd addnn ; "
    command += "{} serve ".format(addnn_executable)
    command += "--controller-host={} ".format(controller_host)
    command += "--controller-port={} ".format(controller_port)
    command += "--placement={} ".format(placement_type)
    command += "--input-shape={} ".format(",".join([str(dimension) for dimension in input_shape]))
    command += "{}".format(scheduler_config.model_path)
    stdout = execute_remote_command(scheduler_config.host, scheduler_config.user, command,
                                    scheduler_config.ssh_key_path)
    print(stdout)


def start_scheduler(scheduler_config: SchedulerConfig, repeat_interval: int, controller_host: str, controller_port: int,
                    placement_type: str, input_shape: Iterable[int]) -> None:
    print("start scheduler at {}".format(scheduler_config.host))

    addnn_executable = "/home/ubuntu/.poetry/bin/poetry run addnn"  # TODO

    command = "cd addnn ; "
    command += "nohup {} serve ".format(addnn_executable)
    command += "--controller-host={} ".format(controller_host)
    command += "--controller-port={} ".format(controller_port)
    command += "--repeat-interval={} ".format(repeat_interval)
    command += "--placement={} ".format(placement_type)
    command += "--input-shape={} ".format(",".join([str(dimension) for dimension in input_shape]))
    command += "{} ".format(scheduler_config.model_path)
    command += "&> /home/ubuntu/scheduler-log.txt & echo $! > ~/scheduler.pid"
    execute_remote_command(scheduler_config.host, scheduler_config.user, command, scheduler_config.ssh_key_path)
