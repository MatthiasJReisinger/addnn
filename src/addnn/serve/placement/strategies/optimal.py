import pulp
import torch
from addnn.controller.proto.controller_pb2 import Node
from addnn.profile.layer_profile import LayerProfile
from addnn.serve.placement.placement import NodeIndex, Placement, get_throughput_matrix, get_latency_matrix
from addnn.serve.placement.strategy import Strategy
from typing import Dict, List, Optional

LayerIndex = int
TierIndex = int


def get_solver() -> Optional[str]:
    available_solvers = pulp.list_solvers(onlyAvailable=True)
    preferred_solvers = ["CPLEX_PY", "CPLEX_CMD"]
    for preferred_solver in preferred_solvers:
        if preferred_solver in available_solvers:
            return pulp.get_solver(preferred_solver)
    return None


class OptimalStrategy(Strategy):
    def name(self) -> str:
        return "optimal"

    def compute_placement(self, nodes: List[Node], layers: List[LayerProfile]) -> Placement:
        problem = pulp.LpProblem("layer-placement", pulp.LpMinimize)

        throughput_matrix = get_throughput_matrix(nodes)
        latency_matrix = get_latency_matrix(nodes)
        node_indices = list(range(len(nodes)))
        layer_indices = list(range(len(layers)))

        placements = pulp.LpVariable.dicts(name="placements",
                                           indexs=(node_indices, layer_indices),
                                           lowBound=0,
                                           upBound=1,
                                           cat=pulp.LpInteger)

        split_points = pulp.LpVariable.dicts(name="split_points",
                                             indexs=(node_indices, node_indices, layer_indices),
                                             lowBound=0,
                                             upBound=1,
                                             cat=pulp.LpInteger)

        num_tiers = max([node.tier for node in nodes]) + 1
        tiers = list(range(0, num_tiers))
        tier_nodes: List[List[NodeIndex]] = [[] for tier in tiers]
        for node_index in node_indices:
            node = nodes[node_index]
            tier_nodes[node.tier].append(node_index)

        compute_latency = pulp.lpSum([
            layers[layer_index].execution_probability * placements[node_index][layer_index] *
            float(layers[layer_index].flops) / float(nodes[node_index].state.resource_state.compute)
            for layer_index in layer_indices for node_index in node_indices
        ])

        communication_latency = pulp.lpSum([
            layers[layer_index].execution_probability *
            split_points[node_index][neighbour_node_index][layer_index - 1] *
            (8 * float(layers[layer_index].marshalled_input_size) /
             float(throughput_matrix[node_index][nodes[neighbour_node_index].host]) +
             latency_matrix[node_index][nodes[neighbour_node_index].host] / 1000.0) for layer_index in layer_indices[1:]
            for node_index in node_indices for neighbour_node_index in _neighbours(node_index, node_indices)
        ])

        input_node_index = 0
        for node_index in range(len(nodes)):
            if nodes[node_index].is_input:
                input_node_index = node_index
                break

        input_latency = pulp.lpSum([
            placements[node_index][0] * (8 * float(layers[0].marshalled_input_size) /
                                         float(throughput_matrix[input_node_index][nodes[node_index].host]) +
                                         latency_matrix[input_node_index][nodes[node_index].host] / 1000.0)
            for node_index in _neighbours(input_node_index, node_indices)
        ])

        # the optimization objective
        problem += compute_latency + communication_latency + input_latency, "End-to-end Latency"

        # constraint: storage capacity of nodes must not be exceeded
        for node_index in node_indices:
            problem += pulp.lpSum([
                placements[node_index][layer_index] * layers[layer_index].storage_size for layer_index in layer_indices
            ]) <= nodes[node_index].state.resource_state.storage

        # constraint: RAM capacity of nodes must not be exceeded
        for node_index in node_indices:
            problem += pulp.lpSum([
                placements[node_index][layer_index] * layers[layer_index].in_memory_size
                for layer_index in layer_indices
            ]) <= nodes[node_index].state.resource_state.memory

        # constraint: each layer has to be assigned to exactly one node
        for layer_index in layer_indices:
            problem += pulp.lpSum([placements[node_index][layer_index] for node_index in node_indices]) == 1.0

        # constraint: non-consecutive layers must not be hosted by the same node
        for node_index in node_indices:
            for layer_index in range(1, len(layers) - 1):
                predecessor_layer_index = layer_index - 1
                successor_layer_index = layer_index + 1
                problem += placements[node_index][predecessor_layer_index] + placements[node_index][
                    successor_layer_index] - 1 <= placements[node_index][layer_index]

        # constraint: a layer's successor cannot be assigned to an earlier tier
        for tier in tiers:
            for layer_index in layer_indices[0:-1]:
                successor_layer_index = layer_index + 1
                successor_tier_node_indices = []
                for successor_tier in tiers[tier:]:
                    successor_tier_node_indices.extend(tier_nodes[successor_tier])

                is_layer_placed_in_current_tier = pulp.lpSum(
                    [placements[node_index][layer_index] for node_index in tier_nodes[tier]])
                is_successor_layer_placed_in_successor_tier = pulp.lpSum(
                    [placements[node_index][successor_layer_index] for node_index in successor_tier_node_indices])

                # if the given layer is placed in the given tier, then the next layer has to be placed either in the current tier on in one of the next tiers
                problem += is_layer_placed_in_current_tier <= is_successor_layer_placed_in_successor_tier

        # constraint: there cannot be any split point at the final layer
        for node_index in node_indices:
            for neighbour_node_index in _neighbours(node_index, node_indices):
                problem += split_points[node_index][neighbour_node_index][layer_indices[-1]] == 0

        # constraint: split points only exist between two different compute nodes:
        for node_index in node_indices:
            for layer_index in layer_indices:
                problem += split_points[node_index][node_index][layer_index] == 0

        # constraint: there is a split point at a node if and only if the node hosts a layer but does not host its successor
        for node_index in node_indices:
            for neighbour_node_index in _neighbours(node_index, node_indices):
                for layer_index in layer_indices[0:-1]:
                    successor_layer_index = layer_index + 1
                    problem += placements[node_index][layer_index] >= split_points[node_index][neighbour_node_index][
                        layer_index]
                    problem += placements[neighbour_node_index][successor_layer_index] >= split_points[node_index][
                        neighbour_node_index][layer_index]
                    problem += placements[node_index][layer_index] + placements[neighbour_node_index][
                        successor_layer_index] - 1 <= split_points[node_index][neighbour_node_index][layer_index]

        solver = get_solver()
        status = problem.solve(solver)

        if status is not pulp.LpStatusOptimal:
            raise Exception("could not find optimal layer-to-node mapping: {}".format(pulp.LpStatus[status]))

        mapping = [-1] * len(layers)
        for node_index in node_indices:
            for layer_index in layer_indices:
                if placements[node_index][layer_index].value() > 0.5:
                    mapping[layer_index] = node_index

        return mapping


def _neighbours(node_index: NodeIndex, node_indices: List[NodeIndex]) -> List[NodeIndex]:
    neighbours = list(node_indices[0:node_index]) + list(node_indices[node_index + 1:])
    return neighbours
