import numpy as np
import os
import pickle as pkl
from time import time

from cascade_generator import si, ic, observe_cascade
from helpers import TimeoutError
from inf_helpers import infection_probability_shortcut
from graph_helpers import filter_graph_by_edges


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

    
def one_run(g, edge_weights, input_path, output_dir):
    obs, c = pkl.load(open(input_path, 'rb'))
    inf_probas = infection_probability_shortcut(g, edge_weights=edge_weights, obs=obs)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    basename = os.path.basename(input_path)
    pkl.dump({'inf_probas': inf_probas},
             open(os.path.join(output_dir, basename), 'wb'))
