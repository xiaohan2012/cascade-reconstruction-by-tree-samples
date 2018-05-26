import pytest
import numpy as np
from graph_tool import Graph
from graph_helpers import extract_nodes, extract_edges
from cascade_generator import get_infection_time, ic, observe_cascade
from numpy.testing import assert_array_equal
from fixture import line, tree



@pytest.fixture
def p(line):
    p = line.new_edge_property('float')
    p.set_value(1.0)
    return p


def test_get_infection_time(line):
    time, edge_list = get_infection_time(line, 0, return_edges=True)
    assert set(edge_list) == {(0, 1), (1, 2), (2, 3)}
    assert_array_equal(time, [0, 1, 2, 3])


@pytest.mark.parametrize('stop_fraction, expected_time, expected_edges',
                         [
                             (0.5, [0, 1, -1, -1], [(0, 1)]),
                             (1.0, [0, 1,  2,  3], [(0, 1), (1, 2), (2, 3)])])
def test_ic(line, p, stop_fraction, expected_time, expected_edges):
    source, time, tree_edges = ic(line, p, 0,
                                  stop_fraction=stop_fraction,
                                  return_tree_edges=True)

    assert source == 0
    assert_array_equal(time, expected_time)

    assert tree_edges == expected_edges


@pytest.fixture
def tree1():
    g = Graph(directed=True)
    g.add_vertex(5)  # one remaining singleton
    g.add_edge_list([(0, 1), (1, 2), (1, 3)])

    # to test 4 is not included
    vfilt = g.new_vertex_property('bool')
    vfilt.set_value(True)
    vfilt[4] = False
    g.set_vertex_filter(vfilt)
    return g
