import array
import functools
import numpy as np
import deap.algorithms
import deap.base
import deap.creator
import deap.tools
import random
from addnn.controller.proto.controller_pb2 import Node
from addnn.profile.layer_profile import LayerProfile
from addnn.serve.placement.placement import NodeIndex, Placement, estimate_ent_to_end_latency, get_throughput_matrix, get_latency_matrix
from addnn.serve.placement.strategy import Strategy
from typing import Dict, List


class GeneticStrategy(Strategy):
    """
    Strategy that employs a basic genetic algorithm to compute a near-optimal placement.
    """
    def name(self) -> str:
        return "genetic"

    def compute_placement(self, nodes: List[Node], layers: List[LayerProfile]) -> Placement:
        throughput_matrix = get_throughput_matrix(nodes)
        latency_matrix = get_latency_matrix(nodes)

        deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0, ))
        deap.creator.create("Individual", array.array, typecode="d", fitness=deap.creator.FitnessMin)

        toolbox = deap.base.Toolbox()
        toolbox.register("node_index", random.randint, 0, len(nodes) - 1)
        toolbox.register("individual",
                         deap.tools.initRepeat,
                         deap.creator.Individual,
                         toolbox.node_index,
                         n=len(layers))
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

        penalty = _penalty(nodes, layers, throughput_matrix, latency_matrix)
        fitness = functools.partial(_fitness,
                                    nodes=nodes,
                                    layers=layers,
                                    throughput_matrix=throughput_matrix,
                                    latency_matrix=latency_matrix,
                                    penalty=penalty)
        toolbox.register("evaluate", lambda individual: (fitness(individual), ))
        toolbox.register("select", deap.tools.selTournament, tournsize=3)
        toolbox.register("mate", deap.tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", deap.tools.mutUniformInt, low=0, up=(len(nodes) - 1), indpb=0.5)

        population_size = 1000
        population = toolbox.population(n=population_size)
        crossover_probability = 0.8
        mutation_probability = 0.02
        generations = 250

        halloffame = deap.tools.HallOfFame(population_size * 0.2)

        result = geneticAlgorithm(
            nodes,
            population,
            toolbox,
            cxpb=crossover_probability,
            mutpb=mutation_probability,
            ngen=generations,
            halloffame=halloffame,
        )
        placement = halloffame[0]

        if not _meets_constraints(placement, nodes, layers):
            raise Exception("could not find valid placement")

        return [int(node_index) for node_index in placement]


def geneticAlgorithm(nodes, population, toolbox, cxpb: float, mutpb: float, ngen: int, halloffame):  # type: ignore
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    unchanged = 0

    # Begin the generational process
    for gen in range(ngen):
        print("generation {}".format(gen))

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - len(halloffame.items))

        # Vary the pool of individuals
        offspring = deap.algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Enfore recursion constraints
        tier_fixed_offspring = []
        for chromosome in offspring:
            _enforce_tier_constraint(nodes, chromosome)
            _enforce_adjacency_constraint(chromosome)
            tier_fixed_offspring.append(chromosome)

        offspring = tier_fixed_offspring

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)

        old_best_fitness = halloffame[0].fitness.values[0]

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        new_best_fitness = halloffame[0].fitness.values[0]

        # Replace the current population by the offspring
        population[:] = offspring

        if new_best_fitness == old_best_fitness:
            unchanged += 1
        else:
            unchanged = 0

        # stop if fitness hasn't improved for a given number of generations
        if unchanged > 10:
            break

    return population


def _penalty(nodes: List[Node], layers: List[LayerProfile], throughput_matrix: List[Dict[str, int]],
             latency_matrix: List[Dict[str, float]]) -> float:
    compute_penalty = float(max([layer.flops for layer in layers])) / float(
        min([node.state.resource_state.compute for node in nodes]))
    minimal_throughput = min([
        throughput_matrix[node_index0][nodes[node_index1].host] for node_index0 in range(len(nodes))
        for node_index1 in _neighbours(node_index0, list(range(len(nodes))))
    ])
    maximal_latency = max([
        latency_matrix[node_index0][nodes[node_index1].host] for node_index0 in range(len(nodes))
        for node_index1 in _neighbours(node_index0, list(range(len(nodes))))
    ])
    communication_penalty = float(max([layer.marshalled_input_size
                                       for layer in layers])) / float(minimal_throughput) + maximal_latency / 1000.0
    return len(layers) * (compute_penalty + communication_penalty)


def _neighbours(node_index: NodeIndex, node_indices: List[NodeIndex]) -> List[NodeIndex]:
    neighbours = list(node_indices[0:node_index]) + list(node_indices[node_index + 1:])
    return neighbours


def _fitness(placement: Placement, nodes: List[Node], layers: List[LayerProfile],
             throughput_matrix: List[Dict[str, int]], latency_matrix: List[Dict[str, float]], penalty: float) -> float:
    fitness = estimate_ent_to_end_latency(placement, nodes, layers, throughput_matrix, latency_matrix)

    if not _meets_constraints(placement, nodes, layers):
        fitness += penalty

    return fitness


def _meets_constraints(placement: Placement, nodes: List[Node], layers: List[LayerProfile]) -> bool:
    # constraint: storage capacity of nodes must not be exceeded
    storage_usage = np.zeros(len(nodes))
    for layer_index in range(len(layers)):
        layer = layers[layer_index]
        storage_usage[int(placement[layer_index])] += layer.storage_size

    for node_index in range(len(nodes)):
        node = nodes[node_index]
        if node.state.resource_state.storage < storage_usage[node_index]:
            return False

    # constraint: RAM capacity of nodes must not be exceeded
    memory_usage = np.zeros(len(nodes))
    for layer_index in range(len(layers)):
        layer = layers[layer_index]
        memory_usage[int(placement[layer_index])] += layer.in_memory_size

    for node_index in range(len(nodes)):
        node = nodes[node_index]
        if node.state.resource_state.memory < memory_usage[node_index]:
            return False

    # constraint: non-consecutive layers must not be hosted by the same node
    assigned_nodes = set()
    for layer_index in range(1, len(layers)):
        if placement[layer_index] != placement[layer_index - 1] and placement[layer_index] in assigned_nodes:
            return False
        assigned_nodes.add(placement[layer_index])

    # constraint: a layer's successor cannot be assigned to an earlier tier
    for layer_index in range(1, len(layers)):
        if nodes[int(placement[layer_index])].tier < nodes[int(placement[layer_index - 1])].tier:
            return False

    return True


def _enforce_tier_constraint(nodes: List[Node], chromosome: Placement) -> None:
    for layer_index in range(1, len(chromosome)):
        previous_node = int(chromosome[layer_index - 1])
        current_node = int(chromosome[layer_index])

        # enforce that a layer's successor cannot be assigned to an earlier tier
        if current_node != previous_node and nodes[current_node].tier < nodes[previous_node].tier:
            chromosome[layer_index] = previous_node


def _enforce_adjacency_constraint(chromosome: Placement) -> None:
    new_ends = dict()
    for layer_index in range(0, len(chromosome)):
        current_node = int(chromosome[layer_index])
        new_ends[current_node] = layer_index

    layer_index = 0
    while layer_index < len(chromosome):
        current_node = int(chromosome[layer_index])
        new_end = new_ends[current_node]
        while layer_index <= new_end:
            chromosome[layer_index] = current_node
            layer_index += 1
