from sample_pool import TreeSamplePool
from tree_stat import TreeBasedStatistics
from random_steiner_tree.util import from_gt
from inference import infection_probability
from graph_tool.centrality import pagerank


def infection_probability_shortcut(g, edge_weights,
                                   obs,
                                   n_samples=1000,
                                   sampling_method='loop_erased'):

    gi = from_gt(g, edge_weights)
    sampler = TreeSamplePool(g, n_samples, sampling_method, gi)
    sampler.fill(obs)
    est = TreeBasedStatistics(g)

    return infection_probability(g, obs, sampler=sampler, error_estimator=est)


def pagerank_scores(g, obs, weight=None, eps=0.0):
    pers = g.new_vertex_property('float')
    pers.a += eps  # add some noise

    for o in obs:
        pers.a[o] = 1

    pers.a /= pers.a.sum()
    rank = pagerank(g, pers=pers, weight=weight)

    if rank.a.sum() == 0:
        raise ValueError('PageRank score all zero')

    p = rank.a / rank.a.sum()
    return p
