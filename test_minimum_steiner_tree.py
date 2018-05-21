import numpy as np
from graph_tool.generation import lattice
from minimum_steiner_tree import min_steiner_tree
from graph_helpers import is_tree


def test_main():
    g = lattice((5, 5))

    for n, c in [(0, 1), (10, 5), (g.num_vertices(), 1)]:
        for _ in range(c):
            obs = np.random.choice(np.arange(g.num_vertices()), g.num_vertices(),
                                   replace=False)
            t = min_steiner_tree(g, obs)
            assert is_tree(t)
            assert set(obs).issubset(set(map(int, t.vertices())))
            
