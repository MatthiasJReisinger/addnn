import yaml
from typing import List, Optional, TextIO


class NodeConfig:
    def __init__(self, host: str, port: int, user: str, compute_capacity: int, tier: int, is_input: bool,
                 network_device: str, ssh_key_path: Optional[str], addnn_executable: str):
        self._host = host
        self._port = port
        self._user = user
        self._compute_capacity = compute_capacity
        self._tier = tier
        self._is_input = is_input
        self._network_device = network_device
        self._ssh_key_path = ssh_key_path
        self._addnn_executable = addnn_executable

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def user(self) -> str:
        return self._user

    @property
    def compute_capacity(self) -> int:
        return self._compute_capacity

    @property
    def tier(self) -> int:
        return self._tier

    @property
    def is_input(self) -> bool:
        return self._is_input

    @property
    def network_device(self) -> str:
        return self._network_device

    @property
    def ssh_key_path(self) -> Optional[str]:
        return self._ssh_key_path

    @property
    def addnn_executable(self) -> str:
        return self._addnn_executable


class SchedulerConfig:
    def __init__(self, host: str, user: str, ssh_key_path: Optional[str], model_path: str) -> None:
        self._host = host
        self._user = user
        self._ssh_key_path = ssh_key_path
        self._model_path = model_path

    @property
    def host(self) -> str:
        return self._host

    @property
    def user(self) -> str:
        return self._user

    @property
    def ssh_key_path(self) -> Optional[str]:
        return self._ssh_key_path

    @property
    def model_path(self) -> str:
        return self._model_path


class BenchmarkConfig:
    def __init__(self, controller_host: str, controller_port: int, start_controller: bool,
                 controller_user: Optional[str], controller_ssh_key: Optional[str],
                 scheduler_config: Optional[SchedulerConfig], dataset_name: str, dataset_root: str,
                 benchmark_duration: int, num_layers: int, node_configs: List[NodeConfig], result_dir: str,
                 seed: Optional[int]) -> None:
        self._controller_host = controller_host
        self._controller_port = controller_port
        self._start_controller = start_controller
        self._controller_user = controller_user
        self._controller_ssh_key = controller_ssh_key
        self._scheduler_config = scheduler_config
        self._dataset_name = dataset_name
        self._dataset_root = dataset_root
        self._benchmark_duration = benchmark_duration
        self._num_layers = num_layers
        self._node_configs = node_configs
        self._result_dir = result_dir
        self._seed = seed

    @property
    def controller_host(self) -> str:
        return self._controller_host

    @property
    def controller_port(self) -> int:
        return self._controller_port

    @property
    def start_controller(self) -> bool:
        return self._start_controller

    @property
    def controller_user(self) -> Optional[str]:
        return self._controller_user

    @property
    def controller_ssh_key(self) -> Optional[str]:
        return self._controller_ssh_key

    @property
    def scheduler_config(self) -> Optional[SchedulerConfig]:
        return self._scheduler_config

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_root(self) -> str:
        return self._dataset_root

    @property
    def benchmark_duration(self) -> int:
        return self._benchmark_duration

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def node_configs(self) -> List[NodeConfig]:
        return self._node_configs

    @property
    def result_dir(self) -> str:
        return self._result_dir

    @property
    def seed(self) -> Optional[int]:
        return self._seed


def read_node_configs(config: TextIO) -> List[NodeConfig]:
    node_configs: List[NodeConfig] = []

    yaml_node_configs = yaml.load(config, yaml.Loader)
    for yaml_node_config in yaml_node_configs["nodes"]:
        ssh_key_path = None
        if "ssh_key_path" in yaml_node_config:
            ssh_key_path = yaml_node_config["ssh_key_path"]

        node_config = NodeConfig(host=yaml_node_config["host"],
                                 port=yaml_node_config["port"],
                                 user=yaml_node_config["user"],
                                 compute_capacity=yaml_node_config["compute_capacity"],
                                 tier=yaml_node_config["tier"],
                                 is_input=yaml_node_config["is_input"],
                                 network_device=yaml_node_config["network_device"],
                                 ssh_key_path=ssh_key_path,
                                 addnn_executable=yaml_node_config["addnn_executable"])
        node_configs.append(node_config)
    return node_configs
