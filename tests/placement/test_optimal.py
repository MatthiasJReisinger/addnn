import pytest
from addnn.controller.proto import controller_pb2
from addnn.node.proto import node_state_pb2
from addnn.profile.layer_profile import LayerProfile
from addnn.serve.placement.strategies.optimal import OptimalStrategy


def test_optimal_placement_of_single_layer_on_single_node() -> None:
    # given
    node = controller_pb2.Node()
    node.host = "127.0.0.1"
    node.port = 42
    node.tier = 0
    node.is_input = True
    node.state.resource_state.memory = 1
    node.state.resource_state.storage = 1
    node.state.resource_state.compute = 1
    node.state.resource_state.bandwidth = 8
    nodes = [node]

    layer = LayerProfile(flops=1,
                         in_memory_size=1,
                         storage_size=1,
                         marshalled_input_size=1,
                         has_exit=False,
                         number_of_exited_samples=0,
                         exit_probability=1.0,
                         execution_probability=1.0,
                         operators=[])
    layers = [layer]

    # when
    strategy = OptimalStrategy()
    placement = strategy.compute_placement(nodes, layers)

    # then
    assert len(placement) == 1
    assert placement[0] == 0


def test_optimal_placement_fails_if_requirements_cannot_be_met() -> None:
    # given
    node = controller_pb2.Node()
    node.host = "127.0.0.1"
    node.port = 42
    node.tier = 0
    node.is_input = True
    node.state.resource_state.memory = 0
    node.state.resource_state.storage = 1
    node.state.resource_state.compute = 1
    node.state.resource_state.bandwidth = 8
    nodes = [node]

    layer = LayerProfile(flops=1,
                         in_memory_size=1,
                         storage_size=1,
                         marshalled_input_size=1,
                         has_exit=False,
                         number_of_exited_samples=0,
                         exit_probability=1.0,
                         execution_probability=1.0,
                         operators=[])
    layers = [layer]

    # when + then
    strategy = OptimalStrategy()
    with pytest.raises(Exception):
        strategy.compute_placement(nodes, layers)


def test_optimal_placement_of_two_layers_on_two_nodes() -> None:
    # given
    node0 = controller_pb2.Node()
    node0.host = "127.0.0.1"
    node0.port = 42
    node0.tier = 0
    node0.is_input = True
    node0.state.resource_state.memory = 1
    node0.state.resource_state.storage = 1
    node0.state.resource_state.compute = 1
    network_throughput = node_state_pb2.NetworkThroughput()
    network_throughput.host = "127.0.0.1"
    network_throughput.throughput = 8
    node0.state.resource_state.network_throughputs.append(network_throughput)
    network_latency = node_state_pb2.NetworkLatency()
    network_latency.host = "127.0.0.1"
    network_latency.latency = 1
    node0.state.resource_state.network_latencies.append(network_latency)

    node1 = controller_pb2.Node()
    node1.CopyFrom(node0)
    node1.is_input = False

    nodes = [node0, node1]

    layer0 = LayerProfile(flops=1,
                          in_memory_size=1,
                          storage_size=1,
                          marshalled_input_size=1,
                          has_exit=False,
                          number_of_exited_samples=0,
                          exit_probability=0.0,
                          execution_probability=1.0,
                          operators=[])
    layer1 = LayerProfile(flops=1,
                          in_memory_size=1,
                          storage_size=1,
                          marshalled_input_size=1,
                          has_exit=False,
                          number_of_exited_samples=0,
                          exit_probability=1.0,
                          execution_probability=1.0,
                          operators=[])
    layers = [layer0, layer1]

    # when
    strategy = OptimalStrategy()
    placement = strategy.compute_placement(nodes, layers)

    # then
    assert len(placement) == 2
    assert placement[0] == 0
    assert placement[1] == 1


def test_optimal_placement_of_two_layers_on_input_node() -> None:
    # given
    node0 = controller_pb2.Node()
    node0.host = "127.0.0.1"
    node0.port = 42
    node0.tier = 0
    node0.is_input = True
    node0.state.resource_state.memory = 2
    node0.state.resource_state.storage = 2
    node0.state.resource_state.compute = 1
    network_throughput = node_state_pb2.NetworkThroughput()
    network_throughput.host = "127.0.0.1"
    network_throughput.throughput = 8
    node0.state.resource_state.network_throughputs.append(network_throughput)
    network_latency = node_state_pb2.NetworkLatency()
    network_latency.host = "127.0.0.1"
    network_latency.latency = 1
    node0.state.resource_state.network_latencies.append(network_latency)

    node1 = controller_pb2.Node()
    node1.CopyFrom(node0)
    node1.is_input = False

    nodes = [node0, node1]

    layer0 = LayerProfile(flops=1,
                          in_memory_size=1,
                          storage_size=1,
                          marshalled_input_size=1,
                          has_exit=False,
                          number_of_exited_samples=0,
                          exit_probability=0.0,
                          execution_probability=1.0,
                          operators=[])
    layer1 = LayerProfile(flops=1,
                          in_memory_size=1,
                          storage_size=1,
                          marshalled_input_size=1,
                          has_exit=False,
                          number_of_exited_samples=0,
                          exit_probability=1.0,
                          execution_probability=1.0,
                          operators=[])
    layers = [layer0, layer1]

    # when
    strategy = OptimalStrategy()
    placement = strategy.compute_placement(nodes, layers)

    # then
    assert len(placement) == 2
    assert placement[0] == 0
    assert placement[1] == 0


def test_optimal_placement_of_multiple_layer_on_multiple_nodes() -> None:
    # given...
    number_of_layers = 4
    number_of_nodes = 4

    # ...nodes with identical resource states
    nodes = []
    for node_index in range(number_of_nodes):
        node = controller_pb2.Node()
        node.host = "{}.{}.{}.{}".format(node_index, node_index, node_index, node_index)
        node.port = 42
        node.tier = 0
        node.is_input = False
        node.state.resource_state.memory = 1
        node.state.resource_state.storage = 1
        node.state.resource_state.compute = 1
        nodes.append(node)

    for node in nodes:
        for neighbour in nodes:
            if node.host != neighbour.host:
                network_throughput = node_state_pb2.NetworkThroughput()
                network_throughput.host = neighbour.host
                network_throughput.throughput = 1_000_000
                node.state.resource_state.network_throughputs.append(network_throughput)

                network_latency = node_state_pb2.NetworkLatency()
                network_latency.host = neighbour.host
                network_latency.latency = 50
                node.state.resource_state.network_latencies.append(network_latency)

    # ...with the first node marked as input source
    nodes[0].is_input = True

    # ...layers with identical resource requirements
    layer = LayerProfile(flops=1,
                         in_memory_size=1,
                         storage_size=1,
                         marshalled_input_size=1,
                         has_exit=False,
                         number_of_exited_samples=0,
                         exit_probability=0.0,
                         execution_probability=1.0,
                         operators=[])
    final_layer = LayerProfile(flops=1,
                               in_memory_size=1,
                               storage_size=1,
                               marshalled_input_size=1,
                               has_exit=False,
                               number_of_exited_samples=0,
                               exit_probability=1.0,
                               execution_probability=1.0,
                               operators=[])
    layers = [layer] * (number_of_layers - 1)
    layers.append(final_layer)

    # when
    strategy = OptimalStrategy()
    placement = strategy.compute_placement(nodes, layers)

    # then
    assert len(placement) == number_of_layers
    assert placement[0] == 0
