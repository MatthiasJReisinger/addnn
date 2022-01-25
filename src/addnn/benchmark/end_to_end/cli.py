import addnn
import addnn.dataset
import click
import os
import shutil
from abc import ABC, abstractmethod
from addnn.benchmark.cli import benchmark
from addnn.benchmark.end_to_end.config import BenchmarkConfig, NodeConfig, SchedulerConfig, read_node_configs
from typing import Any, Dict, Iterable, List, Optional, Tuple, TextIO, TypeVar


@benchmark.group(help="End-to-end system benchmarks.")
@click.option("--controller-host", required=True, help="The host that runs the controller.")
@click.option("--controller-port", required=True, type=int, help="The port at which to reach the controller.")
@click.option("--start-controller",
              is_flag=True,
              default=False,
              help="Whether the controller should be started automatically by the benchmark.")
@click.option("--controller-user", help="The user for the host that runs the controller.")
@click.option("--controller-ssh-key", type=str, help="The path to the ssh key for the controller host.")
@click.option("--start-scheduler",
              is_flag=True,
              default=False,
              help="Whether the scheduler should be started automatically by the benchmark.")
@click.option("--scheduler-host",
              type=str,
              help="If given, the benchmark will automatically start the scheduler at the given host.")
@click.option("--scheduler-user", help="The user for the host that runs the scheduler.")
@click.option("--scheduler-ssh-key", type=str, help="The path to the ssh key for the scheduler host.")
@click.option("--scheduler-model-path", type=str, default="~", help="The path to the model (on the scheduler host).")
@click.option("--dataset",
              "dataset_name",
              required=True,
              type=click.Choice(addnn.dataset.datasets.keys()),
              help="The dataset to use.")
@click.option("--dataset-root",
              default="./datasets",
              show_default=True,
              type=click.Path(exists=True),
              help="The dataset to use.")
@click.option("--benchmark-duration",
              type=int,
              required=False,
              default=10,
              help="The duration of a benchmark interval.")
@click.option("--num-layers",
              type=int,
              required=True,
              help="The number of layers in the model that will be deployed by the scheduler.")
@click.option("--config", required=True, type=click.File())
@click.option("--result-dir",
              default="./benchmark-results",
              show_default=True,
              type=click.Path(),
              help="Path to the directory where benchmark results should be written to.")
@click.option("--seed", type=int, help="The seed to use for the random number generator.")
@click.pass_context
def end_to_end(context: click.Context, controller_host: str, controller_port: int, start_controller: bool,
               controller_user: Optional[str], controller_ssh_key: Optional[str], start_scheduler: bool,
               scheduler_host: Optional[str], scheduler_user: Optional[str], scheduler_ssh_key: str,
               scheduler_model_path: str, dataset_name: str, dataset_root: str, benchmark_duration: int,
               num_layers: int, config: TextIO, result_dir: str, seed: Optional[int]) -> None:
    scheduler_config = None
    if start_scheduler:
        if scheduler_host is None or scheduler_user is None or scheduler_model_path is None:
            raise Exception("incomplete scheduler config")
        scheduler_config = SchedulerConfig(scheduler_host, scheduler_user, scheduler_ssh_key, scheduler_model_path)

    node_configs = read_node_configs(config)

    # initialize the directory for storing the benchmark results
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    benchmark_config = BenchmarkConfig(controller_host, controller_port, start_controller, controller_user,
                                       controller_ssh_key, scheduler_config, dataset_name, dataset_root,
                                       benchmark_duration, num_layers, node_configs, result_dir, seed)
    context.obj = benchmark_config
