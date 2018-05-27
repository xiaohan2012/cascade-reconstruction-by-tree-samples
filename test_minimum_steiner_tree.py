import numpy as np
from graph_tool.generation import lattice
from minimum_steiner_tree import min_steiner_tree
from graph_helpers import is_tree


def test_unweighted():
    g = lattice((5, 5))

    for n, c in [(10, 5), (g.num_vertices(), 1)]:
        # repeat `c` rounds, using `n` terminals
        for _ in range(c):
            obs = np.random.choice(np.arange(g.num_vertices()), n,
                                   replace=False)
            t = min_steiner_tree(g, obs)
            assert is_tree(t)
            assert set(obs).issubset(set(map(int, t.vertices())))


def test_weighted():
    g = lattice((5, 5))

    # assign some random weight
    p = g.new_edge_property('float')
    weights = np.random.random(g.num_edges())
    p.a = (-np.log(weights))
    
    for n, c in [(10, 5), (g.num_vertices(), 1)]:
        # repeat `c` rounds, using `n` terminals
        for _ in range(c):
            obs = np.random.choice(np.arange(g.num_vertices()), n,
                                   replace=False)
            t = min_steiner_tree(g, obs, p)
            # print(t)
            assert is_tree(t)
            assert set(obs).issubset(set(map(int, t.vertices())))
