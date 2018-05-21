import numpy as np
from collections import Counter

from random_steiner_tree import random_steiner_tree

from graph_helpers import swap_end_points, extract_nodes_from_tuples


def tree_probability_gt(out_degree, p_dict, edges, using_log=True):
    if using_log:
        log_numer = np.log([p_dict[(u, v)] for u, v in edges]).sum()
        log_denum = np.log([out_degree[u] for u, _ in edges]).sum()
        return log_numer - log_denum
    else:
        numer = np.product([p_dict[(u, v)] for u, v in edges])
        denum = np.product([out_degree[u] for u, _ in edges])
        assert denum > 0, [out_degree[u] for u, _ in edges]
        return numer / denum


def cascade_probability_gt(g, p_dict, cascade_edges, g_nx, using_log=True):
    """g and g_nx are not used
    """
    if using_log:
        # to prevent floating point underflow
        return np.log([p_dict[(u, v)] for u, v in cascade_edges]).sum()
    else:
        return np.product([p_dict[(u, v)] for u, v in cascade_edges])


def ic_cascade_probability_gt(g, p_dict, cascade_edges, nbr_dict, using_log=True):
    infected_nodes = extract_nodes_from_tuples(cascade_edges)

    if using_log:
        # to prevent floating point underflow
        log_probas_from_active_edges = np.log([p_dict[(u, v)] for u, v in cascade_edges]).sum()
    else:
        probas_from_active_edges = np.product([p_dict[(u, v)] for u, v in cascade_edges])
            
    inactive_edges = {(w, u)  # here it should (w, u) because of graph transpose
                      for u, _ in cascade_edges
                      for w in nbr_dict[u]
                      if w not in infected_nodes}

    inactive_edges -= set(cascade_edges)

    if using_log:
        log_probas_from_inactive_edges = np.log([(1 - p_dict[(u, v)]) for u, v in inactive_edges]).sum()
        return log_probas_from_active_edges + log_probas_from_inactive_edges
    else:
        probas_from_inactive_edges = np.product([(1 - p_dict[(u, v)]) for u, v in inactive_edges])
        return probas_from_active_edges * probas_from_inactive_edges


def tree_probability_nx(g, edges):
    numer = np.product([g[u][v]['weight'] for u, v in edges])
    denum = np.product([g.out_degree(u, weight='weight') for u, v in edges])
    return numer / denum


def casccade_probability_nx(g, cascade_edges):
    return np.product([g[u][v]['weight'] for u, v in cascade_edges])


def sampled_tree_freqs(gi, X, root, sampling_method, N):
    trees = [swap_end_points(random_steiner_tree(gi, X, root, method=sampling_method))
             for i in range(N)]
    tree_freq = Counter(trees)
    return tree_freq
