import pytest
from graph_tool import Graph
from graph_helpers import extract_nodes, extract_edges, get_edge_weights
from preprocess_graph import normalize_globally, preprocess, reverse_edge_weights


@pytest.fixture
def g():
    g = Graph(directed=True)
    g.add_vertex(4)
    g.add_edge_list([(0, 1), (1, 0),
                     (1, 3), (3, 1),
                     (0, 2), (2, 0),
                     (2, 3), (3, 2)])
    weights = g.new_edge_property('float')
    weights[g.edge(0, 1)] = 0.9
    weights[g.edge(1, 0)] = 0.7
    
    weights[g.edge(1, 3)] = 0.8
    weights[g.edge(3, 1)] = 0.2

    weights[g.edge(2, 3)] = 0.4
    weights[g.edge(3, 2)] = 0.3
    
    weights[g.edge(0, 2)] = 0.1
    weights[g.edge(2, 0)] = 0.4
    g.edge_properties['weights'] = weights
    return g


def test_normalize_globally(g):
    norm_g = normalize_globally(g)
    
    assert norm_g.is_directed()

    max_w = 1.5
    expected_edges_and_weights = {
        (0, 1): 0.9 / max_w,
        (1, 0): 0.7 / max_w,
        
        (0, 2): 0.1 / max_w,
        (2, 0): 0.4 / max_w,

        (2, 3): 0.4 / max_w,
        (3, 2): 0.3 / max_w,

        (1, 3): 0.8 / max_w,
        (3, 1): 0.2 / max_w,

        # self-loops
        (0, 0): 0.5 / max_w,
        (1, 1): 0,
        (2, 2): 0.7 / max_w,
        (3, 3): 1.0 / max_w
    }
    assert norm_g.num_edges() == (3 * g.num_vertices())
    assert set(extract_edges(norm_g)) == set(expected_edges_and_weights.keys())
    assert set(extract_nodes(norm_g)) == set(extract_nodes(g))

    new_edge_weights = get_edge_weights(norm_g)
    for (u, v), w in expected_edges_and_weights.items():
        assert pytest.approx(w) == new_edge_weights[norm_g.edge(u, v)]

    deg = norm_g.degree_property_map("out", new_edge_weights)

    for v in norm_g.vertices():
        assert pytest.approx(1.0) == deg[v]


def test_preprocess(g):
    norm_g = preprocess(g)

    max_w = 1.5
    expected_edges_and_weights = {
        (0, 1): 0.7 / max_w,
        (1, 0): 0.9 / max_w,
        
        (0, 2): 0.4 / max_w,
        (2, 0): 0.1 / max_w,

        (2, 3): 0.3 / max_w,
        (3, 2): 0.4 / max_w,

        (1, 3): 0.2 / max_w,
        (3, 1): 0.8 / max_w,

        # self-loops
        (0, 0): 0.5 / max_w,
        (1, 1): 0,
        (2, 2): 0.7 / max_w,
        (3, 3): 1.0 / max_w
    }
    
    new_edge_weights = get_edge_weights(norm_g)
    for (u, v), w in expected_edges_and_weights.items():
        print(u, v)
        assert pytest.approx(w) == new_edge_weights[norm_g.edge(u, v)]


def test_reverse_edge_weights(g):
    g_cp = g.copy()
    g_rev = reverse_edge_weights(g)

    p = get_edge_weights(g_cp)
    p_rev = get_edge_weights(g_rev)

    for e in g_cp.edges():
        u, v = int(e.source()), int(e.target())
        if u < v:
            assert p[g_cp.edge(u, v)] == p_rev[g_rev.edge(v, u)]
