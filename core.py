import random
from graph_helpers import (extract_steiner_tree,
                           gen_random_spanning_tree,
                           filter_graph_by_edges,
                           reachable_node_set,
                           swap_end_points)
from tqdm import tqdm
from random_steiner_tree import random_steiner_tree


def sample_steiner_trees(g, obs,
                         method,
                         n_samples,
                         gi=None,
                         root=None,
                         root_sampler=None,
                         return_type='nodes',
                         log=False,
                         verbose=False):
    """sample `n_samples` steiner trees that span `obs` in `g`

    `method`: the method for sampling steiner tree
    `n_samples`: sample size
    `gi`: the Graph object that is used if `method` in {'cut', 'loop_erased'}
    `root_sampler`: function that samples a root
    `return_type`: if True, return the set of nodes that are in the sampled steiner tree
    """
    assert method in {'cut', 'cut_naive', 'loop_erased'}
    # print('n_samples', n_samples)
    steiner_tree_samples = []
    # for i in tqdm(range(n_samples), total=n_samples):
    if log:
        iters = tqdm(range(n_samples), total=n_samples)
    else:
        iters = range(n_samples)
        
    for i in iters:
        if root is None:
            # if root not give, sample it using some sampler
            if root_sampler is None:
                # print('random root')
                # note: isolated nodes are ignored
                node_set = reachable_node_set(g, list(obs)[0])
                r = int(random.choice(list(node_set)))
            else:
                # print('custom root sampler')
                assert callable(root_sampler), 'root_sampler should be callable'
                # print('root_sampler', root_sampler)
                r = root_sampler()
                # print('root', r)
        else:
            r = root

        if method == 'cut_naive':
            rand_t = gen_random_spanning_tree(g, root=r)
            st = extract_steiner_tree(rand_t, obs, return_nodes=return_type)
            # if return_type:
            #     st = set(map(int, st.vertices()))
        elif method in {'cut', 'loop_erased'}:
            assert gi is not None
            # print('der')
            edges = random_steiner_tree(gi, obs, r, method, verbose=verbose)
            if return_type == 'nodes':
                st = set(u for e in edges for u in e)
            elif return_type == 'tuples':
                st = swap_end_points(edges)
            elif return_type == 'tree':
                st = filter_graph_by_edges(g, edges)
            else:
                raise ValueError('unknown return_type {}'.format(return_type))

        steiner_tree_samples.append(st)

    return steiner_tree_samples
