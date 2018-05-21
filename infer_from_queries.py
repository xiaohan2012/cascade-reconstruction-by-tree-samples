# coding: utf-8

import os
import time
import pickle as pkl
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
from graph_tool import openmp_set_num_threads

from helpers import load_cascades
from inference import infection_probability
from graph_helpers import (load_graph_by_name, remove_filters,
                           observe_uninfected_node, get_edge_weights)
from sample_pool import TreeSamplePool
from random_steiner_tree.util import from_gt, isolate_vertex
from tree_stat import TreeBasedStatistics
from root_sampler import build_root_sampler_by_pagerank_score, build_true_root_sampler


def infer_probas_from_queries(g, obs, c, queries,
                              sampling_method, root_sampler_name, n_samples,
                              with_inc_sampling=False,
                              every=1,
                              iter_callback=None,
                              verbose=False):
    n_nodes = g.num_vertices()

    assert root_sampler_name in {'random', 'pagerank', 'true_root'}

    if root_sampler_name == 'pagerank':
        root_sampler = build_root_sampler_by_pagerank_score(g, obs, c)
    elif root_sampler_name == 'true_root':
        root_sampler = build_true_root_sampler(c)
    else:
        root_sampler = None

    g = remove_filters(g)
    weights = get_edge_weights(g)
    gi = from_gt(g, weights=weights)

    obs_inf = set(obs)
    obs_uninf = set()
    
    probas_list = []

    sampler = TreeSamplePool(g, n_samples=n_samples,
                             method=sampling_method,
                             gi=gi,
                             with_inc_sampling=with_inc_sampling,
                             with_resampling=False,
                             return_type='nodes')
    estimator = TreeBasedStatistics(g)
    sampler.fill(obs,
                 root_sampler=root_sampler)
    estimator.build_matrix(sampler.samples)

    # initial step (without any queries)
    probas = infection_probability(g, obs_inf, sampler, error_estimator=estimator)
    probas_list.append(probas)

    if verbose:
        qs_iter = tqdm(queries)
    else:
        qs_iter = queries
    for i_iter, q in enumerate(qs_iter):
        if c[q] >= 0:  # infected
            obs_inf |= {q}
        else:
            observe_uninfected_node(g, q, obs_inf)
            isolate_vertex(gi, q)
            obs_uninf |= {q}
            # print('g.num_vertices()', g.num_vertices())

        # update samples
        label = int(c[q] >= 0)
        if root_sampler_name == 'pagerank':
            try:
                root_sampler = build_root_sampler_by_pagerank_score(g, obs_inf, c)
            except ValueError:
                print('pagerank score for root_sampler all zero, break')
                break

        if i_iter % every == 0:
            # print('i_iter', i_iter)
            if i_iter == 0:
                node_update_info = {q: label}
            else:
                node_update_info[q] = label

            # evaluate every `every` iteration
            new_samples = sampler.update_samples(obs_inf,
                                                 node_update_info,
                                                 root_sampler=root_sampler,
                                                 log=False,
                                                 verbose=False)
            estimator.update_trees(new_samples, node_update_info)

            # new probas
            probas = infection_probability(g, obs_inf, sampler, error_estimator=estimator)

            # make sure data dimension does not change
            assert len(probas) == n_nodes
            assert g.num_vertices() == n_nodes, '{} != {}'.format(g.num_vertices(), n_nodes)

            probas_list.append(probas)

            # refresh it
            node_update_info = {}

            if callable(iter_callback):
                iter_callback(g, sampler, estimator, obs_inf, obs_uninf)

        else:
            # accumulate info
            node_update_info[q] = label

    return probas_list, sampler, estimator


def one_round(g, obs, c, c_path,
              query_method,
              inference_method,
              query_dirname, inf_proba_dirname,
              n_samples=250,
              every=1,
              root_sampler=None,
              sampling_method='loop_erased',
              with_inc_sampling=False,
              debug=False,
              verbose=False):
    print('\ninference {} started, query_method={}, root_sampler={}, \n'.format(
        c_path, query_method, root_sampler))
    stime = time.time()

    cid = os.path.basename(c_path).split('.')[0]
    probas_dir = inf_proba_dirname
    if not os.path.exists(probas_dir):
        os.makedirs(probas_dir)
    path = os.path.join(probas_dir, '{}.pkl'.format(cid))

    if os.path.exists(path):
        print('{} computed'.format(path))
        return

    query_log_path = os.path.join(query_dirname, '{}.pkl'.format(cid))
    queries, _ = pkl.load(open(query_log_path, 'rb'))

    if inference_method == 'inf_probas':
        # the real part
        probas_list, _, _ = infer_probas_from_queries(g, obs, c, queries,
                                                      sampling_method,
                                                      root_sampler,
                                                      n_samples,
                                                      with_inc_sampling=with_inc_sampling,
                                                      every=every,
                                                      verbose=verbose)
        pkl.dump(probas_list, open(path, 'wb'))
    else:
        raise ValueError('unknown inference method "{}"'.format(inference_method))

    print('\ninference {} done (query_method={}, inference_method={}): taking {:.4f} secs\n'.format(
        c_path,
        query_method, inference_method,
        time.time() - stime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--graph',
                        help='graph name')
    parser.add_argument('-f', '--graph_suffix', required=True,
                        help='suffix of graph name')
    parser.add_argument('-w', '--weighted', action='store_true',
                        help='use weights or not')
    parser.add_argument('-s', '--n_samples', type=int,
                        default=100,
                        help='number of samples')
    parser.add_argument('-m', '--inference_method',
                        default='inf_probas',
                        choices=('inf_probas'),
                        help='method used for infer hidden infections')
    parser.add_argument('--sampling_method',
                        default='loop_erased',
                        choices=('loop_erased', 'cut'),
                        help='')

    parser.add_argument('-i', '--with_inc_sampling',
                        action='store_true',
                        help='whether with incremental sampling or not')
    parser.add_argument('--query_method',
                        help='query method used for infer hidden infections')
    parser.add_argument('-r', '--root_sampler', type=str,
                        default='pagerank',
                        choices={'pagerank', 'random', 'true_root'},
                        help='the steiner tree sampling method')

    parser.add_argument('-c', '--cascade_dir',
                        help='directory to read cascades')
    parser.add_argument('-q', '--query_dirname',
                        required=True,
                        help='directory of queries')
    parser.add_argument('-p', '--inf_proba_dirname',
                        required=True,
                        help='directory to store the inferred probabilities')
    parser.add_argument('--eval_every',
                        default=1,
                        type=int,
                        help='evaluate every ?')
    parser.add_argument('-j', '--n_jobs', type=int, default=-1,
                        help='number of workers in parallel')
    parser.add_argument('--debug',
                        action='store_true',
                        help='')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='')

    args = parser.parse_args()

    print("Args:")
    print('-' * 10)
    for k, v in args._get_kwargs():
        print("{}={}".format(k, v))

    graph_name = args.graph
    graph_suffix = args.graph_suffix
    n_samples = args.n_samples

    query_dirname = args.query_dirname
    inf_proba_dirname = args.inf_proba_dirname

    g = load_graph_by_name(graph_name, weighted=args.weighted,
                           suffix=graph_suffix)

    cascades = load_cascades(args.cascade_dir)

    if not args.debug:
        openmp_set_num_threads(1)  # prevent jobjib from hanging
        Parallel(n_jobs=args.n_jobs)(delayed(one_round)(
            g, tpl[0], tpl[1], path, args.query_method,
            args.inference_method,
            query_dirname,
            inf_proba_dirname, n_samples=n_samples,
            root_sampler=args.root_sampler,
            with_inc_sampling=args.with_inc_sampling,
            sampling_method=args.sampling_method,
            every=args.eval_every,
            verbose=args.verbose)
                                     for path, tpl in tqdm(cascades))
    else:
        for path, tpl in tqdm(cascades):
            one_round(g, tpl[0], tpl[1], path, args.query_method,
                      args.inference_method,
                      query_dirname,
                      inf_proba_dirname, n_samples=n_samples,
                      root_sampler=args.root_sampler,
                      with_inc_sampling=args.with_inc_sampling,
                      every=args.eval_every,
                      sampling_method=args.sampling_method,
                      verbose=args.verbose)
