import addnn.logging
import grpc
import icmplib
import json
import logging
import multiprocessing
import psutil
import statistics
import subprocess
import time
from addnn.node.proto import node_pb2, node_pb2_grpc, node_state_pb2
from addnn.controller.proto import controller_pb2
from google.protobuf.empty_pb2 import Empty
from typing import Dict, Iterable, List, Optional, Any

logger = logging.getLogger(__name__)


class NodeStateMonitor(multiprocessing.Process):
    """
    Monitors the local state of the compute node.
    """
    def __init__(self, initial_resource_state: node_state_pb2.ResourceState, network_monitor_interval: int,
                 iperf_time: int, uds_address: str, model_cache_path: str, node_runtime_pid: int):
        super().__init__()
        self._network_monitor_interval = network_monitor_interval
        self._iperf_time = iperf_time
        self._initial_resource_state = initial_resource_state
        self._uds_address = uds_address
        self._model_cache_path = model_cache_path
        self._node_runtime_process = psutil.Process(pid=node_runtime_pid)
        self._last_monitored_time: Dict[str, float] = {}
        self._last_monitored_throughput: Dict[str, float] = {}

    def run(self) -> None:
        addnn.logging.init("node_state_monitor.log")

        channel = grpc.insecure_channel(self._uds_address)
        self._node_stub = node_pb2_grpc.NodeStub(channel)

        while True:
            self._monitor()
            time.sleep(10)

    def _monitor(self) -> None:
        try:
            neighbour_nodes = self._node_stub.ReadNeighbourNodes(Empty()).neighbour_nodes

            current_resource_state = self._get_current_resource_state(neighbour_nodes)
            logger.debug("current resource state:\n{}".format(current_resource_state))

            # update the state at the node service
            update_resource_state_request = node_pb2.UpdateResourceStateRequest()
            update_resource_state_request.resource_state.CopyFrom(current_resource_state)
            self._node_stub.UpdateResourceState(update_resource_state_request)
        except Exception as e:
            logger.error("could not monitor node state (reason: {})".format(e))

    def _get_current_resource_state(self,
                                    neighbour_nodes: Iterable[controller_pb2.Node]) -> node_state_pb2.ResourceState:

        resource_state = node_state_pb2.ResourceState()

        # TODO ignore the RAM usage of the layers that are already in-memory
        resource_state.memory = psutil.virtual_memory().available

        node_runtime_cpu_percent = self._node_runtime_process.cpu_percent() / psutil.cpu_count()
        used_cpu_percent = psutil.cpu_percent() - node_runtime_cpu_percent

        # since node_runtime_cpu_percent and used_cpu_percent aren't measured at _exactly_ the same time it may be
        # possible that node_runtime_cpu_percent > used_cpu_percent, therefore round up to 0 in that case
        used_cpu_percent = max(0, used_cpu_percent)

        # compute the theoretically available compute power of the node's host, ignoring the CPU usage of the node
        # runtime itself
        resource_state.compute = int(self._initial_resource_state.compute * (100.0 - used_cpu_percent) / 100.0)

        resource_state.bandwidth = self._initial_resource_state.bandwidth

        resource_state.storage = psutil.disk_usage(self._model_cache_path).free

        self._monitor_network_throughput(neighbour_nodes, resource_state)
        _monitor_network_latency(neighbour_nodes, resource_state)

        return resource_state

    def _monitor_network_throughput(self, neighbour_nodes: Iterable[controller_pb2.Node],
                                    resource_state: node_state_pb2.ResourceState) -> None:
        for neighbour_node in neighbour_nodes:
            if neighbour_node.host in self._last_monitored_time and time.time() - self._last_monitored_time[
                    neighbour_node.host] < self._network_monitor_interval:
                logger.debug("skip network througput monitoring for \n{}".format(neighbour_node.host))
                network_throughput = node_state_pb2.NetworkThroughput()
                network_throughput.host = neighbour_node.host
                network_throughput.throughput = int(self._last_monitored_throughput[neighbour_node.host])
                resource_state.network_throughputs.append(network_throughput)
                continue

            logger.debug("monitor network throughput for \n{}".format(neighbour_node.host))

            iperf_command = [
                "iperf3", "--json", "-c", neighbour_node.host, "-p",
                str(neighbour_node.iperf_port), "-t", "{}".format(self._iperf_time)
            ]
            completed_process = subprocess.run(iperf_command, capture_output=True)

            if completed_process.returncode == 0:
                json_output = json.loads(completed_process.stdout)
                throughput = float(json_output["end"]["sum_sent"]["bits_per_second"])

                network_throughput = node_state_pb2.NetworkThroughput()
                network_throughput.host = neighbour_node.host
                network_throughput.throughput = int(throughput)
                resource_state.network_throughputs.append(network_throughput)
                self._last_monitored_time[neighbour_node.host] = time.time()
                self._last_monitored_throughput[neighbour_node.host] = throughput


def _monitor_network_latency(neighbour_nodes: Iterable[controller_pb2.Node],
                             resource_state: node_state_pb2.ResourceState) -> None:
    neighbour_hosts = [neighbour_node.host for neighbour_node in neighbour_nodes]
    if len(neighbour_hosts) > 0:
        ping_results = icmplib.multiping(addresses=neighbour_hosts, count=5, privileged=False)
        for ping_result in ping_results:
            network_latency = node_state_pb2.NetworkLatency()
            network_latency.host = ping_result.address
            network_latency.latency = statistics.median(ping_result.rtts)
            resource_state.network_latencies.append(network_latency)
