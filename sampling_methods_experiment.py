# coding: utf-8

import argparse
# import pandas as pd
import numpy as np
from graph_tool import openmp_set_num_threads
import pickle as pkl
from tqdm import tqdm
from copy import copy

from joblib import delayed, Parallel

from experiment import gen_input
from helpers import infected_nodes
from graph_helpers import load_graph_by_name, get_edge_weights
from core import sample_steiner_trees
from tree_stat import TreeBasedStatistics
from random_steiner_tree.util import from_gt

from root_sampler import build_root_sampler_by_pagerank_score
# from sklearn.metrics import average_precision_score


def incremental_simulation(g, c, p, return_new_edges=False):
    visited = {v: False for v in np.arange(g.num_vertices())}
    new_c = copy(c)
    for v in infected_nodes(c):
        visited[v] = True

    if return_new_edges:
        new_edges = []
        
    queue = list(infected_nodes(c))
    while len(queue) > 0:
        u = queue.pop(0)
        uu = g.vertex(u)
        for e in uu.out_edges():
            v = int(e.target())
            if np.random.random() <= p[e] and not visited[v]:  # active
                if return_new_edges:
                    new_edges.append((u, v))
                new_c[v] = c[u] + 1
                visited[v] = True
                queue.append(v)

    if return_new_edges:
        return (new_c, new_edges)
    else:
        return new_c


def one_run(g, norm_g, q, eps, root_sampler_name, min_size, max_size,
            observation_method="uniform",
            with_inc=False):
    print("observation_method", observation_method)

    n_samples = 100

    p = g.edge_properties['weights']

    obs, c = gen_input(
        g, source=None,
        p=p, q=q,
        model='ic',
        observation_method=observation_method,
        min_size=min_size,
        max_size=max_size)

    print('cascade size', len(infected_nodes(c)))
    # inf_nodes = infected_nodes(c)
    source = np.nonzero(c == 0)[0][0]

    if root_sampler_name == 'pagerank':
        root_sampler = build_root_sampler_by_pagerank_score(g, obs, c, eps=eps)
    elif root_sampler_name == 'true':
        root_sampler = (lambda: source)
    else:
        root_sampler = (lambda: None)

    # method 2:
    # vanilla steiner tree sampling
    gi = from_gt(norm_g, weights=get_edge_weights(norm_g))
    st_tree_nodes = sample_steiner_trees(g, obs, root=root_sampler(),
                                         method='cut', n_samples=n_samples,
                                         gi=gi, return_tree_nodes=True)
    node_stat = TreeBasedStatistics(g, st_tree_nodes)
    st_naive_probas = node_stat.unconditional_proba()

    if with_inc:
        # method 3
        # with incremental cascade simulation
        st_tree_nodes = sample_steiner_trees(g, obs, root=root_sampler(),
                                             method='cut', n_samples=n_samples, gi=gi,
                                             return_tree_nodes=True)
        new_tree_nodes = []
        for nodes in st_tree_nodes:
            fake_c = np.ones(g.num_vertices()) * (-1)
            fake_c[list(nodes)] = 1
            new_c = incremental_simulation(g, fake_c, p, return_new_edges=False)
            new_tree_nodes.append(infected_nodes(new_c))
        node_stat = TreeBasedStatistics(g, new_tree_nodes)
        st_tree_inc_probas = node_stat.unconditional_proba()
            
        # y_true = np.zeros((len(c), ))
        # y_true[inf_nodes] = 1

        # mask = np.array([(i not in obs) for i in range(len(c))])

    row = {
        'c': c,
        'obs': obs,
        'st_naive_probas': st_naive_probas
    }

    if with_inc:
        row['st_tree_inc_probas'] = st_tree_inc_probas
    # # for inf_probas in [brute_force_inf_probas, st_naive_probas, st_tree_inc_probas]:
    # for inf_probas in [st_naive_probas, st_tree_inc_probas]:
    #     row.append(average_precision_score(y_true[mask], inf_probas[mask]))
    return row


if __name__ == '__main__':
    openmp_set_num_threads(1)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--graph', help='graph name')
    parser.add_argument('-f', '--graph_suffix',
                        required=True,
                        help='suffix of graph name')
    parser.add_argument('-n', '--n_runs',
                        type=int,
                        help='num. of runs')
    parser.add_argument('--min_size',
                        type=int,
                        help='minimum cascade size')
    parser.add_argument('--max_size',
                        type=int,
                        help='maximum cascade size')
    
    parser.add_argument('-q', '--obs_fraction',
                        type=float,
                        help='fraction of observed infections')
    parser.add_argument('--observation_method',
                        type=str,
                        help='observation method')
    parser.add_argument('-o', '--output_path',
                        help='output_path')

    args = parser.parse_args()

    print("Args:")
    print('-' * 10)
    for k, v in args._get_kwargs():
        print("{}={}".format(k, v))

    graph_name = args.graph
    suffix = args.graph_suffix
    n_runs = args.n_runs
    q = args.obs_fraction
    observation_method = args.observation_method
    min_size = args.min_size
    max_size = args.max_size
    
    g = load_graph_by_name(graph_name, weighted=True)
    norm_g = load_graph_by_name(graph_name, weighted=True, suffix=suffix)

    print('g.num_edges()', g.num_edges())
    print('norm_g.num_edges()', norm_g.num_edges())
    
    result = {}
    # if False:
    for eps in [0.0, 0.5]:
        rows = Parallel(n_jobs=-1)(delayed(one_run)(g, norm_g, q, eps, 'pagerank',
                                                    min_size, max_size,
                                                    observation_method=observation_method)
                                   for i in tqdm(range(n_runs), total=n_runs))
        # df = pd.DataFrame(rows, columns=['st_vanilla', 'st_inc'])
        print('pagerank, eps=', eps)
        # print(df.describe())
        result['pagerank-eps{}'.format(eps)] = rows

    print('root sampler = None')
    rows = Parallel(n_jobs=-1)(delayed(one_run)(g, norm_g, q, 0.0, None,
                                                min_size, max_size,
                                                observation_method=observation_method)
                               for i in tqdm(range(n_runs), total=n_runs))
    # df = pd.DataFrame(rows, columns=['st_vanilla', 'st_inc'])
    # print(df.describe())
    result['random_root'] = rows

    print('root sampler = real source')
    rows = Parallel(n_jobs=-1)(delayed(one_run)(g, norm_g, q, 0.0, 'true',
                                                min_size, max_size,
                                                observation_method=observation_method)
                               for i in tqdm(range(n_runs), total=n_runs))
    # df = pd.DataFrame(rows, columns=['st_vanilla', 'st_inc'])
    # print(df.describe())
    result['true_root'] = rows

    pkl.dump(result, open(args.output_path, 'wb'))
