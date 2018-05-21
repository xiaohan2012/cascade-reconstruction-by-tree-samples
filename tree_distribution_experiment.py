# coding: utf-8

import networkx as nx
import numpy as np
import pandas as pd
import pickle as pkl
import random

from scipy.spatial.distance import cosine, cdist
from tqdm import tqdm
from itertools import combinations
from joblib import Parallel, delayed
from graph_tool import openmp_set_num_threads

from random_steiner_tree.util import from_nx

from proba_helpers import sampled_tree_freqs, tree_probability_nx as tree_probability


HIGH = 1
LOW = 0.0


def one_run(num_vertices, size_X, n_samples, low=LOW, high=HIGH):
    g = nx.complete_graph(num_vertices, create_using=nx.DiGraph())
    
    for u, v in g.edges_iter():
        g[u][v]['weight'] = (high - low) * np.random.random() + low

    gi = from_nx(g)

    if True:
        X = np.random.permutation(g.number_of_nodes())[:size_X]
        root = random.choice(list(set(g.nodes()) - set(X)))
    else:
        # root infects terminals
        X, root = min(g.edges_iter(), key=lambda e: g[e[0]][e[1]]['weight'])
        X = [X]

    # print('root = {}'.format(root))
    # print('terminals = {}'.format(X))
    # print(g[X[0]][root]['weight'])

    lerw_tree_freq = sampled_tree_freqs(gi, X, root, 'loop_erased', n_samples)

    cut_tree_freq = sampled_tree_freqs(gi, X, root, 'cut', n_samples)
    
    all_trees = set(lerw_tree_freq.keys()) | set(cut_tree_freq.keys())
    # print("num. unique trees: {}".format(len(all_trees)))

    lerw_actual_probas = np.array([lerw_tree_freq[t] / n_samples for t in all_trees])
    cut_actual_probas = np.array([cut_tree_freq[t] / n_samples for t in all_trees])

    expected_probas = np.array([tree_probability(g, t)
                                for t in all_trees])
    expected_probas /= expected_probas.sum()

    name_and_probas = [('True', expected_probas),
                       ('lerw', lerw_actual_probas),
                       ('cut', cut_actual_probas)]

    cosine_sims = {}
    abs_dists = {}
    for (n1, p1), (n2, p2) in combinations(name_and_probas, 2):
        # print('sim({}, {})={}'.format(n1, n2, 1-cosine(p1, p2)))
        cosine_sims[(n1, n2)] = 1-cosine(p1, p2)
        abs_dists[(n1, n2)] = cdist([p1], [p2], 'minkowski', p=1.0)[0, 0]
    return cosine_sims, abs_dists


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

    num_vertices = args.num_vertices
    size_x = args.num_terminals
    n_samples = args.n_samples
    n_runs = args.n_runs

    openmp_set_num_threads(1)
    
    records = Parallel(n_jobs=4)(
        delayed(one_run)(num_vertices, size_x, n_samples=n_samples, low=LOW, high=HIGH)
        for i in tqdm(range(n_runs), total=n_runs))
    cosine_records, abs_records = zip(*records)

    cosine_df = pd.DataFrame.from_records(filter(None, cosine_records))
    abs_df = pd.DataFrame.from_records(filter(None, abs_records))
    
    data = {
        'cosine_sim': cosine_df.describe(),
        'l1_dist': abs_df.describe()
    }
    for k, d in data.items():
        print(k)
        print(d)
    pkl.dump(data, open(args.output, 'wb'))
    
