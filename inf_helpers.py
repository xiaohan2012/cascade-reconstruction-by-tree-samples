from sample_pool import TreeSamplePool
from tree_stat import TreeBasedStatistics
from random_steiner_tree.util import from_gt
from inference import infection_probability


def infection_probability_shortcut(g, edge_weights,
                                   obs,
                                   n_samples=1000,
                                   sampling_method='loop_erased'):

    gi = from_gt(g, edge_weights)
    sampler = TreeSamplePool(g, n_samples, sampling_method, gi)
    sampler.fill(obs)
    est = TreeBasedStatistics(g)

    return infection_probability(g, obs, sampler=sampler, error_estimator=est)
