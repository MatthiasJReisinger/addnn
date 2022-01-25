import pytest
from addnn.controller.proto.controller_pb2 import Node
from addnn.serve.placement.strategies.genetic import _enforce_tier_constraint, _enforce_adjacency_constraint


def test_enforce_tier_constraint_does_not_change_valid_chromosome() -> None:
    # given
    node0 = Node()
    node0.tier = 0
    node1 = Node()
    node1.tier = 1
    nodes = [node0, node1]
    chromosome = [0, 0, 1, 1]

    # when
    _enforce_tier_constraint(nodes, chromosome)

    # then
    assert (chromosome == [0, 0, 1, 1])


def test_enforce_tier_constraint_fixes_invalid_chromosome() -> None:
    # given
    node0 = Node()
    node0.tier = 0
    node1 = Node()
    node1.tier = 1
    nodes = [node0, node1]
    chromosome = [1, 1, 0, 0]

    # when
    _enforce_tier_constraint(nodes, chromosome)

    # then
    assert (chromosome == [1, 1, 1, 1])


def test_enforce_adjacency_constraint_fixes_invalid_chromosome() -> None:
    # given
    chromosome = [0, 2, 1, 1, 2]

    # when
    _enforce_adjacency_constraint(chromosome)

    # then
    assert (chromosome == [0, 2, 2, 2, 2])


def test_enforce_adjacency_constraint_fixes_invalid_chromosome_with_multiple_constraint_violations() -> None:
    # given
    chromosome = [0, 2, 1, 1, 2, 4, 3, 4]

    # when
    _enforce_adjacency_constraint(chromosome)

    # then
    assert (chromosome == [0, 2, 2, 2, 2, 4, 4, 4])


def test_enforce_adjacency_constraint_does_not_change_valid_chromosome() -> None:
    # given
    chromosome = [0, 2, 2, 2, 2]

    # when
    _enforce_adjacency_constraint(chromosome)

    # then
    assert (chromosome == [0, 2, 2, 2, 2])
