import numpy as np
import pickle as pkl
from tqdm import tqdm
from time import time

from cascade_generator import si, ic, observe_cascade
from query_selection import EntropyQueryGenerator
from inference import infection_probability
from helpers import TimeoutError
from eval_helpers import top_k_infection_precision_recall
from graph_helpers import (isolate_node, remove_filters,
                           filter_graph_by_edges,
                           hide_disconnected_components,
                           load_graph_by_name,
                           extract_edges)


def gen_input(g, source=None, cascade_path=None, stop_fraction=0.25, p=0.5, q=0.1, model='si',
              observation_method='uniform',
              min_size=10, max_size=100,
              return_tree=False):
    # print('observation_method', observation_method)
    tree_requiring_methods = {'leaves', 'bfs-head', 'bfs-tail'}

    if cascade_path is None:
        if model == 'si':
            s, c, tree = si(g, p, stop_fraction=stop_fraction,
                            source=source)
        elif model == 'ic':
            start = time()
            while True:
                if time() - start > 1:
                    # re-try another root
                    raise TimeoutError()

                s, c, tree_edges = ic(g, p, source=source,
                                      min_size=min_size,
                                      max_size=max_size,
                                      return_tree_edges=(observation_method in tree_requiring_methods))
                size = np.sum(c >= 0)
                if size >= min_size and size <= max_size:  # size fits
                    # print('big enough')
                    # do this later because it's slow
                    if return_tree:
                        tree = filter_graph_by_edges(g, tree_edges)
                        print('tree.is_directed()', tree.is_directed())
                        print('tree.num_vertices()', tree.num_vertices())
                        print('tree.num_edges()', tree.num_edges())
                    else:
                        tree = None
                    # print('source', s)
                    # print('tree.edges()', extract_edges(tree))
                    break
                # print('{} not in range ({}, {})'.format(size, min_size, max_size))
        else:
            raise ValueError('unknown cascade model')
    else:
        print('load from cache')
        c = pkl.load(open(cascade_path, 'rb'))
        s = np.nonzero([c == 0])[1][0]
        
    obs = observe_cascade(c, s, q, observation_method, tree=tree)
    # print(obs)
    if not return_tree:
        return obs, c, None
    else:
        return obs, c, tree


def gen_inputs_varying_obs(
        g, source=None, cascade_path=None, stop_fraction=0.25, p=0.5, q=0.1, model='si',
        observation_method='uniform',
        min_size=10, max_size=100,
        n_times=8,
        return_tree=False):
    """return a bunch of sampled inputs given the same cascade simulation
    for speed-up and result statbility
    """
    # print('observation_method', observation_method)
    tree_requiring_methods = {'leaves', 'bfs-head', 'bfs-tail'}

    if cascade_path is None:
        while True:
            try:
                if model == 'si':
                    s, c, tree = si(g, p, stop_fraction=stop_fraction,
                                    source=source)
                elif model == 'ic':
                    start = time()
                    while True:
                        # time out, change root
                        if time() - start > 3:
                            raise TimeoutError()

                        s, c, tree_edges = ic(
                            g, p, source=source,
                            min_size=min_size,
                            max_size=max_size,
                            return_tree_edges=(observation_method in tree_requiring_methods))
                        size = np.sum(c >= 0)
                        if size >= min_size and size <= max_size:  # size fits
                            # print('big enough')
                            # do this later because it's slow
                            if return_tree:
                                tree = filter_graph_by_edges(g, tree_edges)
                                print('tree.is_directed()', tree.is_directed())
                                print('tree.num_vertices()', tree.num_vertices())
                                print('tree.num_edges()', tree.num_edges())
                            else:
                                tree = None
                            # print('source', s)
                            # print('tree.edges()', extract_edges(tree))
                            break
                        # print('{} not in range ({}, {})'.format(size, min_size, max_size))
                else:
                    raise ValueError('unknown cascade model')
                break
            except TimeoutError:
                print('timeout')
                continue
    else:
        print('load from cache')
        _, c, tree = pkl.load(open(cascade_path, 'rb'))
        s = np.nonzero([c == 0])[1][0]
    
    for i in range(n_times):
        obs = observe_cascade(c, s, q, observation_method, tree=tree)
        if not return_tree:
            yield obs, c, None
        else:
            yield obs, c, tree

    
# @profile
def one_round_experiment(g, obs, c, q_gen, query_method, ks,
                         inference_method='sampling',
                         n_spanning_tree_samples=100,
                         subset_size=None,
                         n_queries=10,
                         return_details=False,
                         debug=False,
                         log=False):
    """
    str query_method: {'random', 'ours', 'pagerank}
    inference_method: {'min_steiner_tree', 'sampling'}
    ks: k values in top-`k` for evaluation

    return_details: bool, whether queries should be teturned

    Return:

    dict: k -> [(precision, recall), ...]
    """
    if not debug:
        # if debug, we need to check how the graph is changed
        g = remove_filters(g)  # protect the original graph

    # assert not g.is_directed()
    
    performance = {k: [] for k in ks}  # grouped by k values
    inf_nodes = list(obs)

    queries = []
     
    if log:
        iters = tqdm(range(n_queries), total=n_queries)
    else:
        iters = range(n_queries)

    for i in iters:
        if query_method in {'random', 'pagerank'}:
            q = q_gen.select_query()
        elif query_method == 'ours':
            q = q_gen.select_query(g, inf_nodes)
        else:
            raise ValueError('no such method..')

        if debug:
            print('query: {}'.format(q))

        queries.append(q)

        if c[q] == -1:
            # uninfected
            # remove it from graph
            # ignore for now
            # filter the node will change the vertex index
            isolate_node(g, q)
            hide_disconnected_components(g, inf_nodes)
            q_gen.update_pool(g)
        else:
            inf_nodes.append(q)

        if inference_method == 'sampling':
            probas = infection_probability(g, inf_nodes,
                                           n_samples=n_spanning_tree_samples,
                                           subset_size=subset_size)
        else:
            print('try {} later'.format(inference_method))

        for k in ks:
            scores = top_k_infection_precision_recall(
                g, probas, c, obs, k)
            performance[k].append(scores)

    if return_details:
        return performance, queries
    else:
        return performance

if __name__ == '__main__':
    g = load_graph_by_name('karate')
    obs, c = gen_input(g)
    our_gen = EntropyQueryGenerator(remove_filters(g), obs, num_spt=100, num_stt=5, use_resample=True)
    score = one_round_experiment(g, obs, c, our_gen, 'ours', 10, log=True)
    print(score)
