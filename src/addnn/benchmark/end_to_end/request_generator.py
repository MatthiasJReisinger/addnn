import addnn
import addnn.grpc
import grpc
import multiprocessing
import pickle
import random
import torch
import time
from addnn.controller.proto.controller_pb2_grpc import ControllerStub
from addnn.controller.proto import controller_pb2, controller_pb2_grpc
from addnn.dataset import Dataset
from addnn.node.proto.neural_network_pb2 import InferRequest
from addnn.node.proto.neural_network_pb2_grpc import NeuralNetworkStub
from google.protobuf.empty_pb2 import Empty
from typing import Any, Dict, Iterable, List, Optional, Tuple, TextIO, TypeVar

WARM_UP_DURATION = 10


class InferenceBenchmark:
    def __init__(self, request_time: int, response_time: int) -> None:
        self._request_time = request_time
        self._response_time = response_time

    @property
    def request_time(self) -> int:
        """
        Timestamp at which the request was sent (unix time stamp in nanoseconds).
        """
        return self._request_time

    @property
    def response_time(self) -> int:
        """
        Timestamp at which the response was received (unix time stamp in nanoseconds).
        """
        return self._response_time


def generate_requests_for_duration_in_multiple_processes(controller_host: str, controller_port: int, dataset: Dataset,
                                                         dataset_root: str,
                                                         benchmark_duration: int) -> List[InferenceBenchmark]:
    number_of_benchmark_processes = 1
    print("start generating requests")

    benchmark_processes: List[multiprocessing.Process] = []
    result_queues: List[multiprocessing.Queue] = []

    benchmark_start_event = multiprocessing.Event()
    benchmark_stop_event = multiprocessing.Event()

    for _ in range(number_of_benchmark_processes):
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        result_queues.append(result_queue)
        benchmark_process = multiprocessing.Process(target=generate_requests_until_stoped,
                                                    args=(result_queue, benchmark_start_event, benchmark_stop_event,
                                                          controller_host, controller_port, dataset, dataset_root,
                                                          benchmark_duration))
        benchmark_process.start()
        benchmark_processes.append(benchmark_process)

    time.sleep(WARM_UP_DURATION)
    benchmark_start_event.set()

    time.sleep(benchmark_duration)
    benchmark_stop_event.set()

    for benchmark_process in benchmark_processes:
        benchmark_process.join()

    inference_benchmarks: List[InferenceBenchmark] = []

    for result_queue in result_queues:
        inference_benchmarks.extend(result_queue.get())

    return inference_benchmarks


# multiprocessing.Event is only a factory function and does not name a type, so this defines a dummy alias for mypy
Event = Any


def generate_requests_until_stoped(result_queue: multiprocessing.Queue, benchmark_start_event: Event,
                                   benchmark_stop_event: Event, controller_host: str, controller_port: int,
                                   dataset: Dataset, dataset_root: str) -> None:
    train_loader, test_loader = dataset.load(dataset_root, batch_size=1, num_workers=0)

    controller_stub = addnn.grpc.create_tcp_stub(ControllerStub, controller_host, controller_port)
    input_node = addnn.util.controller.get_input_node(controller_stub)

    neural_network_stub = addnn.grpc.create_tcp_stub(NeuralNetworkStub, input_node.host, input_node.port)

    inference_benchmarks: List[InferenceBenchmark] = []

    stop = False
    while not stop:
        for batch, target in test_loader:
            infer_request = InferRequest()
            infer_request.tensor = pickle.dumps(batch)
            try:
                infer_response = neural_network_stub.Infer(infer_request)
                print("latency: {}".format((infer_response.end_timestamp - infer_response.start_timestamp) / 1000**3))

                if benchmark_start_event.is_set():
                    inference_benchmark = InferenceBenchmark(infer_response.start_timestamp,
                                                             infer_response.end_timestamp)
                    inference_benchmarks.append(inference_benchmark)
            except:
                print("dropped request")

            if benchmark_stop_event.is_set():
                stop = True
                break

    result_queue.put(inference_benchmarks)


WARM_UP_REQUESTS = 5


def generate_requests_in_multiple_processes(controller_host: str, controller_port: int, dataset: Dataset,
                                            dataset_root: str, number_of_requests: int,
                                            seed: Optional[int]) -> List[InferenceBenchmark]:
    number_of_benchmark_processes = 1
    print("start generating requests")

    benchmark_processes: List[multiprocessing.Process] = []
    result_queues: List[multiprocessing.Queue] = []

    benchmark_start_event = multiprocessing.Event()
    benchmark_stop_event = multiprocessing.Event()

    for _ in range(number_of_benchmark_processes):
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        result_queues.append(result_queue)
        benchmark_process = multiprocessing.Process(target=_generate_requests,
                                                    args=(result_queue, controller_host, controller_port, dataset,
                                                          dataset_root, number_of_requests, seed))
        benchmark_process.start()
        benchmark_processes.append(benchmark_process)

    for benchmark_process in benchmark_processes:
        benchmark_process.join()

    inference_benchmarks: List[InferenceBenchmark] = []

    for result_queue in result_queues:
        inference_benchmarks.extend(result_queue.get())

    return inference_benchmarks


def _generate_requests(result_queue: multiprocessing.Queue, controller_host: str, controller_port: int,
                       dataset: Dataset, dataset_root: str, number_of_requests: int, seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)

    train_loader, test_loader = dataset.load(dataset_root, batch_size=1, num_workers=0)

    controller_stub = addnn.grpc.create_tcp_stub(ControllerStub, controller_host, controller_port)
    input_node = addnn.util.controller.get_input_node(controller_stub)

    neural_network_stub = addnn.grpc.create_tcp_stub(NeuralNetworkStub, input_node.host, input_node.port)

    inference_benchmarks: List[InferenceBenchmark] = []

    number_of_sent_requests = 0
    stop = False
    while not stop:
        for batch, target in test_loader:
            if number_of_sent_requests >= WARM_UP_REQUESTS + number_of_requests:
                stop = True
                break

            infer_request = InferRequest()
            infer_request.tensor = pickle.dumps(batch)
            infer_response = neural_network_stub.Infer(infer_request)
            print("latency: {}".format((infer_response.end_timestamp - infer_response.start_timestamp) / 1000**3))

            if number_of_sent_requests > WARM_UP_REQUESTS:
                inference_benchmark = InferenceBenchmark(infer_response.start_timestamp, infer_response.end_timestamp)
                inference_benchmarks.append(inference_benchmark)

            number_of_sent_requests += 1

    result_queue.put(inference_benchmarks)
