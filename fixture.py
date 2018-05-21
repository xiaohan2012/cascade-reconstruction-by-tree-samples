import numpy as np
import pytest

from graph_tool import Graph
from graph_tool.generation import lattice
from random_steiner_tree import util
from graph_helpers import remove_filters, get_edge_weights
from preprocess_graph import preprocess


@pytest.fixture
def g():
    graph = remove_filters(lattice((10, 10)))
    
    graph.set_directed(True)
    edges_iter = list(graph.edges())
    for e in edges_iter:
        graph.add_edge(e.target(), e.source())
        
    ew = graph.new_edge_property('float')
    ew.a = np.random.random(graph.num_edges()) * 0.2 + 0.8
    graph.edge_properties['weights'] = ew

    return preprocess(graph)


@pytest.fixture
def obs(g):
    return np.random.choice(np.arange(g.num_vertices()), 10, replace=False)


@pytest.fixture
def gi(g):
    return util.from_gt(g, get_edge_weights(g))


@pytest.fixture
def tree():
    g = Graph(directed=True)
    g.add_vertex(4)
    g.add_edge_list([(0, 1), (1, 2), (1, 3)])
    return g


@pytest.fixture
def line():
    g = Graph(directed=True)
    g.add_vertex(4)
    g.add_edge_list([(0, 1), (1, 2), (2, 3)])
    return g
