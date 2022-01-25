import addnn.benchmark.strategies.memory_monitor
import addnn.serve.cli
import multiprocessing
import os
import psutil
import random
import resource
import subprocess
import time
from addnn.controller.proto import controller_pb2
from addnn.profile.layer_profile import LayerProfile
from addnn.serve.placement import strategy_loader
from addnn.serve.placement.placement import Placement
from typing import List


class BenchmarkResult:
    """
    Represents the result of a single benchmark run.
    """
    def __init__(self, placement: Placement, duration: float, memory_usage: int):
        self._placement = placement
        self._duration = duration
        self._memory_usage = memory_usage

    @property
    def placement(self) -> Placement:
        return self._placement

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def memory_usage(self) -> int:
        return self._memory_usage


class Executor(multiprocessing.Process):
    """
    Executes a benchmark run.
    """
    def __init__(self, strategy_name: str, nodes: List[controller_pb2.Node], layer_properties: List[LayerProfile],
                 seed: int, result_queue: multiprocessing.Queue):
        super().__init__()
        self._strategy_name = strategy_name
        self._nodes = nodes
        self._layer_properties = layer_properties
        self._seed = seed
        self._result_queue = result_queue

    def run(self) -> None:
        random.seed(self._seed)

        max_memory = multiprocessing.Value("i")
        max_memory.value = 0
        _start_memory_monitor(max_memory)

        strategy = strategy_loader.load_strategy(self._strategy_name)
        placement = strategy.compute_placement(self._nodes, self._layer_properties)
        resource_usage = resource.getrusage(resource.RUSAGE_SELF)
        child_resource_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        duration = resource_usage.ru_utime + resource_usage.ru_stime + child_resource_usage.ru_utime + child_resource_usage.ru_stime

        # get the amount of memory that was used by the strategy
        with max_memory.get_lock():
            memory_usage = max_memory.value

        result = BenchmarkResult(placement, duration, memory_usage)
        self._result_queue.put(result)


def _start_memory_monitor(
        max_memory: multiprocessing.sharedctypes.Synchronized
) -> addnn.benchmark.strategies.memory_monitor.MemoryMonitor:
    pid = os.getpid()
    monitor_interval = 1  # TODO make configurable
    memory_monitor = addnn.benchmark.strategies.memory_monitor.MemoryMonitor(pid, monitor_interval, max_memory)
    memory_monitor.daemon = True
    memory_monitor.start()
    return memory_monitor
