import multiprocessing
import multiprocessing.sharedctypes
import psutil
import subprocess
import time


class MemoryMonitor(multiprocessing.Process):
    """
    Monitors the memory usage of the benchmark runs.
    """
    def __init__(self, pid: int, interval: int, max_memory: multiprocessing.sharedctypes.Synchronized) -> None:
        super().__init__()
        self._process = psutil.Process(pid)
        self._interval = interval
        self._max_memory = max_memory

    def run(self) -> None:
        while True:
            self._monitor()
            time.sleep(self._interval)

    def _monitor(self) -> None:
        with self._max_memory.get_lock():
            current_memory_usage = _get_memory_usage(self._process)
            if current_memory_usage > self._max_memory.value:
                self._max_memory.value = current_memory_usage


def _get_memory_usage(process: psutil.Process) -> int:
    current_memory_usage = process.memory_full_info().uss
    for child_process in process.children(recursive=True):
        current_memory_usage += child_process.memory_full_info().uss
    return current_memory_usage
