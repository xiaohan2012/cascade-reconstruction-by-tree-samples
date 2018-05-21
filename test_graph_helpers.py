import pytest
from numpy.testing import assert_almost_equal
from graph_tool import Graph
from graph_tool.topology import label_components
from graph_tool.generation import complete_graph, lattice
from graph_tool.search import bfs_search
from scipy.stats import entropy

from fixture import line, tree


from fixture import g, obs
from graph_helpers import (extract_steiner_tree, filter_graph_by_edges,
                           extract_edges, extract_nodes,
                           remove_filters,
                           contract_graph_by_nodes,
                           isolate_node,
                           hide_disconnected_components,
                           k_hop_neighbors,
                           pagerank_scores,
                           reachable_node_set,
                           get_leaves,
                           BFSNodeCollector, reverse_bfs)



@pytest.mark.parametrize("X,edges", [([1, 3], {(1, 3)}),
                                     ([0, 2], {(0, 1), (1, 2)}),
                                     ([0], set()),
                                     ([0, 1, 2, 3], {(0, 1), (1, 2), (1, 3)})])
def test_extract_steiner_tree(X, edges):
    g = complete_graph(4)
    tree = filter_graph_by_edges(g, [(0, 1), (1, 2), (1, 3)])

    # CASE 1
    # extract **tree**
    stt = extract_steiner_tree(tree, X, return_nodes=False)
    assert set(extract_edges(stt)) == edges

    if len(X) > 1:
        nodes = {u for e in edges for u in e}
        assert set(extract_nodes(stt)) == nodes
    else:
        # single terminal case
        assert set(extract_nodes(stt)) == set(X)

    # CASE 2
    # extract **nodes**
    stt = extract_steiner_tree(tree, X, return_nodes=True)
    if len(X) > 1:
        nodes = {u for e in edges for u in e}
        assert stt == nodes
    else:
        # single terminal case
        assert stt == set(X)


def test_contract_graph_by_nodes():
    def get_weight_by_edges(g, weights, edges):
        return [weights[g.edge(u, v)] for u, v in edges]
    
    # weighted graph
    g = lattice((2, 2))
    weights = g.new_edge_property('float')
    for e in g.edges():
        weights[e] = int(e.target())

    # contract 0 and 1
    cg, new_weights = contract_graph_by_nodes(g, [0, 1], weights)
    assert set(extract_nodes(cg)) == set(range(3))
    edges = [(0, 0), (0, 1), (0, 2), (1, 2)]
    assert set(extract_edges(cg)) == set(edges)
    assert_almost_equal(get_weight_by_edges(cg, new_weights, edges),
                        [1, 2, 3, 3])

    # contract 0, 1 and 2
    cg, new_weights = contract_graph_by_nodes(g, [0, 1, 2], weights)
    assert set(extract_nodes(cg)) == set(range(2))
    edges = [(0, 0), (0, 1)]
    assert set(extract_edges(cg)) == set(edges)
    assert_almost_equal(get_weight_by_edges(cg, new_weights, edges),
                        [3, 6])

    # contract all nodes
    cg, new_weights = contract_graph_by_nodes(g, [0, 1, 2, 3], weights)
    assert set(extract_nodes(cg)) == {0}
    assert set(extract_edges(cg)) == {(0, 0)}
    assert_almost_equal(new_weights.a, [9])

    # contract just 1
    cg, new_weights = contract_graph_by_nodes(g, [0], weights)
    assert set(extract_nodes(cg)) == set(extract_nodes(g))
    assert set(extract_edges(cg)) == set(extract_edges(g))
    assert_almost_equal(new_weights.a, weights.a)


def test_isolate_node():
    g = remove_filters(Graph(directed=True))
    g.add_vertex(4)
    g.add_edge_list([(0, 1), (1, 0), (0, 2,), (2, 0), (0, 3), (3, 0)])

    isolate_node(g, 1)
    assert set(label_components(g, directed=False)[0].a) == {0, 1}

    isolate_node(g, 0)
    assert set(label_components(g, directed=False)[0].a) == {0, 1, 2, 3}


def test_isolate_disconnected_components():
    ######### case 1 #######
    g = remove_filters(Graph(directed=False))
    g.add_vertex(4)
        
    hide_disconnected_components(g, [0, 2])  # 1, 3 are isolated
    assert set(extract_nodes(g)) == {0, 2}

    ######### case 2 #######
    g = remove_filters(Graph(directed=False))
    g.add_vertex(4)
    g.add_edge(0, 1)
    g.add_edge(1, 2)

    hide_disconnected_components(g, [0])
    assert set(extract_nodes(g)) == {0, 1, 2}
    

def test_k_hop_neighbors():
    """
    0 -- 1
    |    |
    2 -- 3 -- 4  -- 5
    """
    g = Graph(directed=False)
    g.add_vertex(6)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
    for u, v in edges:
        g.add_edge(u, v)

    assert k_hop_neighbors(0, g, k=1) == {1, 2}
    assert k_hop_neighbors(0, g, k=2) == {1, 2, 3}
    assert k_hop_neighbors(0, g, k=3) == {1, 2, 3, 4}
    for k in (4, 5, 6, 7, 8, 9, 10):
        assert k_hop_neighbors(0, g, k=k) == {1, 2, 3, 4, 5}
    
    assert k_hop_neighbors(3, g, k=1) == {1, 2, 4}
    assert k_hop_neighbors(3, g, k=2) == {0, 1, 2, 4, 5}


def test_pagerank_scores(g, obs):
    """as eps increases, the more uncertain the pagerank scores should be,
    thus, higher entropy"""
    eps_list = [0.0, 0.25, 0.5, 1.0]
    pr_list = []
    for eps in eps_list:
        pr = pagerank_scores(g, obs, eps)
        pr_list.append(pr)

    for pr1, pr2 in zip(pr_list, pr_list[1:]):
        ent1, ent2 = entropy(pr1), entropy(pr2)
        assert ent1 < ent2


def test_reachable_node_set():
    g = Graph(directed=False)
    g.add_vertex(4)
    g.add_edge_list([(0, 1), (1, 2)])
    actual = reachable_node_set(g, source=0)
    assert actual == {0, 1, 2}


@pytest.mark.parametrize('tree, expected',
                         [(tree(), [2, 3]),
                          (line(), [3])])
def test_get_leaves(tree, expected):
    leaves = get_leaves(tree, deg='out')
    assert list(leaves) == expected


@pytest.mark.parametrize('g, expected',
                         [(line(), [0, 1, 2, 3]),
                          (tree(), [0, 1, 2, 3])])
def test_BFSNodeCollectorVisitor(g, expected):
    vis = BFSNodeCollector()
    bfs_search(g, 0, vis)
    assert vis.nodes_in_order == expected


@pytest.mark.parametrize('g, expected',
                         [(line(), [3, 2, 1, 0]),
                          (tree(), [2, 3, 1, 0])])
def test_reverse_bfs(g, expected):
    nodes = reverse_bfs(g)
    assert nodes == expected
