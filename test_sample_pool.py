import pytest

import numpy as np
from sample_pool import TreeSamplePool
from test_helpers import check_tree_samples
from graph_helpers import observe_uninfected_node, extract_nodes
from random_steiner_tree.util import isolate_vertex
from fixture import g, gi, obs

N_SAMPLES = 1000

# @pytest.mark.parametrize('return_type', ['nodes', 'tuples'])
# def test_resampling(g, gi, obs, return_type):
#     pool = TreeSamplePool(g, gi=gi,
#                           n_samples=N_SAMPLES,
#                           method='loop_erased',
#                           with_inc_sampling=False,
#                           with_resampling=True,
#                           return_type=return_type)
#     pool.fill(obs)

#     if return_type == 'tuples':
#         # type checking
#         for t in pool.samples:
#             assert isinstance(t, tuple)
#             for e in t:
#                 assert isinstance(e, tuple)
#                 assert len(e) == 2
#         unique_resampled_trees = set(pool.samples)

#     elif return_type == 'nodes':
#         for t in pool.samples:
#             assert isinstance(t, set)
#         unique_resampled_trees = set(map(tuple, pool.samples))

#     unique_sampled_trees = set(pool._old_samples)

#     print(len(unique_resampled_trees))
#     assert len(unique_resampled_trees) < 10  # far few unique resampled trees
#     assert len(unique_sampled_trees) == N_SAMPLES  # with high probability

#     if return_type == 'nodes':
#         n_nodes_to_rm = 5
#         qs = np.random.choice(list(set(extract_nodes(g)) - set(obs)), size=n_nodes_to_rm)
#         for i in qs:
#             isolate_vertex(gi, i)
#             observe_uninfected_node(g, i, obs)
#             pool.update_samples(obs, i, 0)

#         c = [0] * g.num_vertices()
#         for q in qs:
#             c[q] = -1

#         check_tree_samples(qs, c, pool.samples)
