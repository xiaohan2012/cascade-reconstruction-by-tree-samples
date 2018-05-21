# coding: utf-8

import networkx as nx
import numpy as np
import random
import pandas as pd
from scipy.spatial.distance import cosine, cdist
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import Counter, OrderedDict
from random_steiner_tree import random_steiner_tree
from random_steiner_tree.util import from_nx

from graph_helpers import swap_end_points
from proba_helpers import casccade_probability_nx, tree_probability_nx, sampled_tree_freqs


def l1_dist(probas1, probas2):
    return cdist([probas1],
                 [probas2],
                 'minkowski', p=1.0)[0, 0]


def one_run(num_vertices, num_terminals, n_samples, sampling_method, low, high):
    g = nx.complete_graph(num_vertices, create_using=nx.DiGraph())

    for u, v in g.edges_iter():
        g[u][v]['weight'] = (high - low) * np.random.random() + low

    X = np.random.permutation(g.number_of_nodes())[:num_terminals]
    root = random.choice(list(set(g.nodes()) - set(X)))

    # now, let's do the adjustment
    # 1. reverse the edge direction
    # 2. compute the tree probability
    # 3. do importance resampling

    # reverse the edge direction
    g_rev = g.copy()
    for u, v in g.edges_iter():
        if u < v:
            g_rev[u][v]['weight'], g_rev[v][u]['weight'] = g_rev[v][u]['weight'], g_rev[u][v]['weight']
    gi_rev = from_nx(g_rev)

    tree_freq_rev = sampled_tree_freqs(gi_rev, X, root, sampling_method, n_samples)
    possible_trees = list(tree_freq_rev.keys())

    tree_probas_rev = np.array([tree_freq_rev[t] for t in possible_trees]) / n_samples
    cascade_probas = np.array([casccade_probability_nx(g_rev, t) for t in possible_trees])
    cascade_probas /= cascade_probas.sum()

    # distance without re-sampling
    cos_sim_rev_only = 1 - cosine(tree_probas_rev, cascade_probas)
    l1_dist_rev_only = l1_dist(tree_probas_rev, cascade_probas)

    # now we do the re-sampling
    trees = [swap_end_points(random_steiner_tree(gi_rev, X, root, method=sampling_method))
             for i in range(n_samples)]
    possible_trees = list(set(trees))

    # caching table
    p_tbl = {t: casccade_probability_nx(g_rev, t) for t in possible_trees}
    pi_tbl = {t: tree_probability_nx(g_rev, t) for t in possible_trees}

    p_T = np.array([p_tbl.get(t, 0) for t in trees])
    pi_T = np.array([pi_tbl.get(t, 0) for t in trees])
    sampling_weights = p_T / pi_T

    sampling_weights /= sampling_weights.sum()  # normlization

    # re-sampling trees by weights
    resampled_tree_idx = np.random.choice(n_samples, p=sampling_weights,
                                          replace=True, size=n_samples)

    resampled_trees = [trees[i] for i in resampled_tree_idx]

    resampled_tree_freq = Counter(resampled_trees)
    resampled_tree_probas = np.array([resampled_tree_freq[t] for t in possible_trees]) / n_samples

    # here we calculate the probas based on g_rev
    # because edges point towards root
    cascade_probas = np.array([casccade_probability_nx(g_rev, t) for t in possible_trees])
    cascade_probas /= cascade_probas.sum()

    cos_sim_together = 1 - cosine(resampled_tree_probas, cascade_probas)
    l1_dist_together = l1_dist(resampled_tree_probas,
                               cascade_probas)
    ans = OrderedDict()
    ans['cos_sim_without_resampling'] = cos_sim_rev_only
    ans['l1_dist_without_resampling'] = l1_dist_rev_only
    ans['cos_sim_with_resampling'] = cos_sim_together
    ans['l1_dist_with_resampling'] = l1_dist_together

    return ans


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--num_vertices',
                        type=int,
                        help='num of vertices of the complete graph')
    parser.add_argument('-x', '--num_terminals',
                        type=int, default=2,
                        help='num  of terminals of the complete graph')
    parser.add_argument('-k', '--n_samples',
                        type=int, default=10000000,
                        help='num  of steiner tree samples')
    parser.add_argument('-r', '--n_runs',
                        type=int, default=48,
                        help='num of runs')
    parser.add_argument('-o', '--output',
                        help='output path')

    args = parser.parse_args()

    print("Args:")
    print('-' * 10)
    for k, v in args._get_kwargs():
        print("{}={}".format(k, v))
    
    high = 1
    low = 0.0
    sampling_method = 'loop_erased'
    num_vertices = args.num_vertices
    num_terminals = args.num_terminals
    n_samples = args.n_samples

    n_runs = args.n_runs

    recs = Parallel(n_jobs=4)(
        delayed(one_run)(num_vertices, num_terminals, n_samples, sampling_method,
                         low, high) for i in tqdm(range(n_runs), total=n_runs))
    df = pd.DataFrame.from_records(recs)

    df.describe().to_pickle(args.output)
