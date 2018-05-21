import pytest
import numpy as np
from query_selection import (RandomQueryGenerator, EntropyQueryGenerator,
                             PRQueryGenerator, PredictionErrorQueryGenerator,
                             MutualInformationQueryGenerator,
                             SamplingBasedGenerator,
                             LatestFirstOracle,
                             EarliestFirstOracle,
                             NoMoreQuery)
from simulator import Simulator
from graph_helpers import remove_filters, get_edge_weights
from fixture import g, line
from sample_pool import TreeSamplePool
from random_steiner_tree.util import from_gt
from tree_stat import TreeBasedStatistics

from test_helpers import check_tree_samples, check_error_esitmator, check_samples_so_far


def iter_callback(g, q_gen, *args):
    if isinstance(q_gen, SamplingBasedGenerator):
        check_samples_so_far(g, q_gen.sampler,
                             q_gen.error_estimator, *args)

            
@pytest.mark.parametrize("query_method", ['random', 'pagerank', 'entropy', 'error', 'mutual-info'])
@pytest.mark.parametrize("sampling_method", ['cut', 'loop_erased'])
@pytest.mark.parametrize("with_inc_sampling", [False])
@pytest.mark.parametrize("root_sampler", ['true_root', 'random'])
def test_query_method(g, query_method, sampling_method, root_sampler, with_inc_sampling):
    print('query_method: ', query_method)
    print('sampling_method: ', sampling_method)
    print('root_sampler: ', root_sampler)

    gv = remove_filters(g)
    print(gv.num_edges())
    edge_weights = get_edge_weights(gv)

    if query_method in {'entropy', 'error', 'mutual-info'}:
        gi = from_gt(g, edge_weights)
    else:
        gi = None

    pool = TreeSamplePool(gv,
                          n_samples=20,
                          method=sampling_method,
                          gi=gi,
                          return_type='nodes',  # using tree nodes
                          with_inc_sampling=with_inc_sampling
    )

    error_estimator = TreeBasedStatistics(gv)

    if query_method == 'random':
        q_gen = RandomQueryGenerator(gv)
    elif query_method == 'pagerank':
        q_gen = PRQueryGenerator(gv)
    elif query_method == 'entropy':
        q_gen = EntropyQueryGenerator(gv, pool,
                                      error_estimator=error_estimator,
                                      root_sampler=root_sampler)
    elif query_method == 'error':
        q_gen = PredictionErrorQueryGenerator(gv, pool,
                                              error_estimator=error_estimator,
                                              prune_nodes=True,
                                              n_node_samples=None,
                                              root_sampler=root_sampler)
    elif query_method == 'mutual-info':
        q_gen = MutualInformationQueryGenerator(
            gv, pool,
            error_estimator=error_estimator,
            prune_nodes=True,
            n_node_samples=None,
            root_sampler=root_sampler)

    sim = Simulator(gv, q_gen, gi=gi, print_log=True)
    print('simulator created')
    n_queries = 10
    qs, aux = sim.run(n_queries,
                      gen_input_kwargs={'min_size': 20},
                      iter_callback=iter_callback)
    print('sim.run finished')

    assert len(qs) == n_queries
    assert set(qs).intersection(set(aux['obs'])) == set()

    if query_method in {'entropy', 'error', 'weighted-prederror'}:
        check_tree_samples(qs, aux['c'], q_gen.sampler.samples)

    if query_method in {'error', 'weighted-prederror'}:
        # ensure that error estimator updates its tree samples
        check_error_esitmator(qs, aux['c'], error_estimator)


def test_no_more_query(g):
    gv = remove_filters(g)

    q_gen = RandomQueryGenerator(gv)
    sim = Simulator(gv, q_gen, print_log=True)

    qs, aux = sim.run(g.num_vertices()+100)
    assert len(qs) < g.num_vertices()
    

def build_simulator_using_prediction_error_query_selector(g, **kwargs):
    gv = remove_filters(g)
    gi = from_gt(g)
    pool = TreeSamplePool(gv,
                          n_samples=1000,
                          method='loop_erased',
                          gi=gi,
                          return_type='nodes'  # using tree nodes
    )

    q_gen = PredictionErrorQueryGenerator(gv, pool,
                                          error_estimator=TreeBasedStatistics(gv),
                                          root_sampler='random',
                                          **kwargs)
    return Simulator(gv, q_gen, gi=gi, print_log=True), q_gen


@pytest.mark.parametrize("repeat_id", range(5))
def test_prediction_error_with_candidate_pruning(g, repeat_id):
    min_probas = [0, 0.1, 0.2, 0.3, 0.4]
    cand_nums = []

    for min_proba in min_probas:
        sim, q_gen = build_simulator_using_prediction_error_query_selector(
            g, prune_nodes=True, min_proba=min_proba)

        sim.run(0)  # just get the candidates
        q_gen.prune_candidates()  # and prune the candidates

        cand_nums.append(len(q_gen._cand_pool))

    # number of candidates should be decreasing (more accurately, non-increasing)
    for prev, cur in zip(cand_nums, cand_nums[1:]):
        assert prev >= cur


def test_prediction_error_sample_nodes_for_estimation(g):
    n_node_samples_list = [10, 20, 30, 40, 50]
    for n_node_samples in n_node_samples_list:
        sim, q_gen = build_simulator_using_prediction_error_query_selector(
            g, n_node_samples=n_node_samples)
        sim.run(0)

        samples = q_gen._sample_nodes_for_estimation()
        assert len(samples) == n_node_samples


@pytest.mark.parametrize("graph, c", [(line(), np.array([0, 1, 2, -1]))])
@pytest.mark.parametrize("method", ['earliest', 'latest'])
def test_oracle_strategy(graph, c, method):
    graph = remove_filters(graph)
    if method == 'earliest':
        q_gen = EarliestFirstOracle(graph)
        expected = [1, 2]
    elif method == 'latest':
        q_gen = LatestFirstOracle(graph)
        expected = [2, 1]

    sim = Simulator(graph, q_gen)
    qs, _ = sim.run(100, obs=[0], c=c)  # will hiddenly raise NoMoreQuery
    
    assert qs == expected
