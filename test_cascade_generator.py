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


@pytest.mark.parametrize('return_tree_edges', [True, False])
def test_ic(line, p, return_tree_edges):
    source, time, tree_edges = ic(line, p, 0, return_tree_edges=return_tree_edges)

    assert source == 0
    assert_array_equal(time, [0, 1, 2, 3])

    if return_tree_edges:
        assert tree_edges == [(0, 1), (1, 2), (2, 3)]
    else:
        assert tree_edges is None


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


@pytest.mark.parametrize('tree, expected',
                         [(line(), [3]),
                          (tree1(), [2, 3])])
def test_observe_cascade_on_leaves(tree, expected):
    c = np.array([0, 1, 2, 3])  # dummy
    obs = observe_cascade(c,
                          None, q=1.0,
                          method='leaves',
                          tree=tree, source_includable=True)
    assert list(obs) == expected


@pytest.mark.parametrize('tree, expected',
                         [(line(), [3]),
                          (tree1(), [2, 3])])
def test_observe_cascade_on_leaves(tree, expected):
    c = np.array([0, 1, 2, 3])  # dummy
    obs = observe_cascade(c,
                          None, q=1.0,
                          method='leaves',
                          tree=tree, source_includable=True)
    assert list(obs) == expected
    


@pytest.mark.parametrize('tree, q, expected',
                         [
                             (line(), 0.75, [0, 1, 2]),
                             (tree(), 0.75, [0, 1, 2]),

                             (line(), 0.5, [0, 1]),
                             (tree(), 0.5, [0, 1])])
def test_observe_bfs_head(tree, q, expected):
    c = np.array([0, 1, 2, 3])  # dummy
    obs = observe_cascade(c,
                          0, q=q,
                          method='bfs-head',
                          tree=tree, source_includable=True)
    assert list(obs) == expected



@pytest.mark.parametrize('tree, q, expected',
                         [
                             (line(), 0.75, [1, 2, 3]),
                             (tree(), 0.75, [1, 2, 3]),

                             (line(), 0.5, [2, 3]),
                             (tree(), 0.5, [2, 3])])
def test_observe_bfs_tail(tree, q, expected):
    c = np.array([0, 1, 2, 3])  # dummy
    obs = observe_cascade(c,
                          0, q=q,
                          method='bfs-tail',
                          tree=tree, source_includable=True)
    assert list(obs) == expected

    
