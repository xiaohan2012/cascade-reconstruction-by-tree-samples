import pytest
import numpy as np
from inference import infer_infected_nodes, infection_probability
from graph_helpers import (gen_random_spanning_tree, extract_nodes,
                           remove_filters,
                           observe_uninfected_node)
from random_steiner_tree.util import isolate_vertex
from sample_pool import TreeSamplePool
from tree_stat import TreeBasedStatistics
from fixture import g, gi, obs


@pytest.mark.parametrize("with_inc_sampling", [True, False])
def test_inf_probas_shape(g, gi, obs, with_inc_sampling):
    """might fail if the removed vertex isolates some observed nodes
    """
    error_estimator = TreeBasedStatistics(g)
    sampler = TreeSamplePool(g, 25, 'cut', gi=gi,
                             return_type='nodes',
                             with_inc_sampling=with_inc_sampling)
    sampler.fill(obs)
    error_estimator.build_matrix(sampler.samples)

    n = g.num_vertices()
    all_nodes = extract_nodes(g)
    remaining_nodes = list(set(all_nodes) - set(obs))

    # remove five nodes
    removed = []
    for i in range(5):
        r = remaining_nodes[i]
        removed.append(r)

        observe_uninfected_node(g, r, obs)
        isolate_vertex(gi, r)

        # update samples
        new_samples = sampler.update_samples(obs, {r: 0})
        error_estimator.update_trees(new_samples, {r: 0})

        # check probas
        probas = error_estimator.unconditional_proba()

        assert probas.shape == (n,)
        for r in removed:
            assert probas[r] == 0
        for o in obs:
            assert probas[o] == 1.0


def test_infer_infected_nodes_sampling_approach(g, gi, obs):
    error_estimator = TreeBasedStatistics(g)
    sampler = TreeSamplePool(g, 100, 'cut', gi=gi,
                             return_type='nodes')
    sampler.fill(obs)
    error_estimator.build_matrix(sampler.samples)

    g = remove_filters(g)

    # with min steiner trees
    inf_nodes = infer_infected_nodes(g, obs, estimator=None, use_proba=False, method="min_steiner_tree")
    # simple test, just make sure observation is in the prediction
    assert set(obs).issubset(set(inf_nodes))

    # sampling approach without probability
    inf_nodes2 = infer_infected_nodes(g, obs, estimator=error_estimator, use_proba=False, method="sampling")
    assert set(obs).issubset(set(inf_nodes2))

    # sampling approach with probability
    probas = infer_infected_nodes(g, obs, estimator=error_estimator, use_proba=True, method="sampling")

    assert isinstance(probas, np.ndarray)
    assert probas.dtype == np.float
