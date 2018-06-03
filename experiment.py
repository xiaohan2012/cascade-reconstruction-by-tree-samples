import numpy as np
import os
import pickle as pkl
from time import time

from cascade_generator import si, ic, observe_cascade, CascadeTooSmall
from helpers import TimeoutError
from inf_helpers import infection_probability_shortcut, pagerank_scores
from graph_helpers import filter_graph_by_edges
from root_sampler import get_root_sampler_by_name


def gen_input(g, source=None, cascade_path=None, stop_fraction=0.25, p=0.5, q=0.1, model='si',
              observation_method='uniform',
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

                try:
                    s, c, tree_edges = ic(g, p, source=source,
                                          stop_fraction=stop_fraction,
                                          return_tree_edges=(observation_method in tree_requiring_methods))
                    if return_tree:
                        tree = filter_graph_by_edges(g, tree_edges)
                    else:
                        tree = None
                    break
                except CascadeTooSmall:
                    continue

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

                        try:
                            s, c, tree_edges = ic(
                                g, p, source=source,
                                stop_fraction=stop_fraction,
                                return_tree_edges=(
                                    (observation_method in tree_requiring_methods)
                                    or return_tree)
                            )
                            if return_tree:
                                tree = filter_graph_by_edges(g, tree_edges)
                            else:
                                tree = None
                            break
                        except CascadeTooSmall as e:
                            # print(str(e))
                            continue
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

    
def one_run(g, edge_weights, input_path, output_dir, method='our',
            **kwargs):
    basename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, basename)

    if os.path.exists(output_path):
        # print(output_path, 'procssed, skip')
        return

    obs, c = pkl.load(open(input_path, 'rb'))

    nlog_edge_weights = g.new_edge_property('float')
    nlog_edge_weights.a = -np.log(edge_weights.a)

    if method == 'our':
        root_sampler_name = kwargs.get('root_sampler_name')
        root_sampler = get_root_sampler_by_name(root_sampler_name, g=g, obs=obs, c=c,
                                                weights=nlog_edge_weights)
        n_samples = kwargs.get('n_sample', 5000)
        inf_probas = infection_probability_shortcut(
            g, edge_weights=edge_weights, obs=obs,
            root_sampler=root_sampler,
            n_samples=n_samples,
            log=False)
    elif method == 'min-steiner-tree':
        from minimum_steiner_tree import min_steiner_tree
        # we want the product of weights, so apply negative log
        # nlog_edge_weights = g.new_edge_property('float')
        # nlog_edge_weights.a = -np.log(edge_weights.a)
        nodes = min_steiner_tree(g, obs,
                                 p=nlog_edge_weights,
                                 return_type='nodes')

        # make it a binary vector
        inf_probas = np.zeros(len(c))
        inf_probas[nodes] = 1
    elif method == 'pagerank':
        # inf_probas is not actually probability
        inf_probas = pagerank_scores(g, obs, weight=edge_weights)
    else:
        raise ValueError('unsupported method')

    pkl.dump({'inf_probas': inf_probas},
             open(output_path, 'wb'))
