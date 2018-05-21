import numpy as np
import networkx as nx
from graph_tool import GraphView

from core import sample_steiner_trees
from graph_helpers import (has_vertex, get_edge_weights, filter_graph_by_edges,
                           extract_nodes_from_tuples)
from proba_helpers import tree_probability_gt, ic_cascade_probability_gt, \
    cascade_probability_gt
from core1 import matching_trees
from helpers import infected_nodes
from cascade_generator import incremental_simulation


class TreeSamplePool():
    def __init__(self, g, n_samples, method,
                 gi=None,
                 with_inc_sampling=False,
                 with_resampling=False,
                 true_casacde_proba_func=ic_cascade_probability_gt,
                 return_type='nodes'):
        assert return_type in {'nodes', 'tuples'}, 'invalid return_type {}'.format(return_type)
        self.g = g
        self.num_nodes = g.num_vertices()  # fixed
        self.n_samples = n_samples
        self.gi = gi
        self.method = method
        self.return_type = return_type
        self.with_inc_sampling = with_inc_sampling
        self._samples = []

        self.true_casacde_proba_func = true_casacde_proba_func
        self.with_resampling = with_resampling
        if self.with_resampling:
            # to enable resampling
            # needs to return edge tuples
            self._internal_return_type = 'tuples'
            self.p = None
            self.g_nx = None
            self.p_dict = None
        else:
            self._internal_return_type = return_type

        # print('DEBUG: TreeSamplePool.with_inc_sampling=', self.with_inc_sampling)

    def fill(self, obs, **kwargs):
        self._samples = sample_steiner_trees(
            self.g, obs,
            method=self.method,
            n_samples=self.n_samples,
            return_type=self._internal_return_type,
            gi=self.gi,
            **kwargs)

        if self.with_inc_sampling:
            self._samples = [self.add_incremental_edges(s)
                             for s in self._samples]

        if self.with_resampling:
            print('DEBUG: TreeSamplePool.with_resampling=', self.with_resampling)
            self._old_samples = self._samples
            self._samples = self.resample_trees(self._samples)

    def add_incremental_edges(self, tree_nodes):
        if isinstance(tree_nodes, GraphView):
            raise TypeError('add_incremental_edges does not support GraphView yet. ' +
                            'Please pass in a set of nodes')
        fake_c = np.ones(self.num_nodes) * (-1)
        fake_c[list(tree_nodes)] = 1

        edge_weights = get_edge_weights(self.g)
        assert edge_weights is not None, 'for incremental edge addition, edge weight should be given'

        new_c = incremental_simulation(self.g, fake_c, edge_weights,
                                       self.num_nodes,
                                       return_new_edges=False)

        return set(infected_nodes(new_c))

    # @profile
    def update_samples(self, inf_nodes, node_update_info, **kwargs):
        """if label=1, assuming `inf_nodes` includes `node` already
        if label=0, assuming `self.g` removes `node` already

        Return:
        new_samples
        """
        for n, label in node_update_info.items():
            assert label in {0, 1}  # 0: uninfected, 1: infected

        if self._internal_return_type == 'tuples':
            def valid(t):
                nodes = extract_nodes_from_tuples(t)
                for n, label in node_update_info.items():
                    if label == 1:
                        if n not in nodes:
                            return False
                    else:
                        if n in nodes:
                            return False
                return True

            valid_samples = [t for t in self._samples if valid(t)]
        elif self.return_type == 'nodes':
            valid_samples = matching_trees(self._samples, node_update_info)

        # print('num. valid_samples: {}'.format(len(valid_samples)))
        new_samples = sample_steiner_trees(
            self.g, inf_nodes,
            method=self.method,
            n_samples=self.n_samples - len(valid_samples),
            return_type=self._internal_return_type,
            gi=self.gi,
            **kwargs)

        if self.with_inc_sampling:
            # print('With incremental sampling')
            new_samples = [self.add_incremental_edges(t)
                           for t in new_samples]

        self._samples = valid_samples + new_samples

        assert len(self._samples) == self.n_samples

        if self.with_resampling:
            self._old_samples = self._samples
            self._samples = self.resample_trees(self._samples)

        # only useful if re-sampling is NOT enabled
        return new_samples

    @property
    def samples(self):
        if not self.with_resampling:
            return self._samples
        else:
            if self.return_type == 'nodes':
                return list(map(extract_nodes_from_tuples, self._samples))
            elif self.return_type == 'tree':
                return [filter_graph_by_edges(self.g, t)
                        for t in self._samples]
            else:
                return self._samples

    @property
    def is_empty(self):
        return len(self._samples) == 0

    def resample_trees(self, trees):
        possible_trees = list(set(trees))

        self.p = get_edge_weights(self.g)

        # this is required for speed
        # graph_tool's out_neighbours is slow
        self.g_nx = nx.DiGraph()
        for e in self.g.edges():
            self.g_nx.add_edge(int(e.source()), int(e.target()))

        self.p_dict = {tuple(map(int, [e.source(), e.target()])): self.p[e]
                       for e in self.g.edges()}

        out_degree = self.g.degree_property_map('out', weight=self.p)
        out_degree_dict = {int(v): out_degree[v] for v in self.g.vertices()}

        # caching table
        # and we work in the log domain
        log_p_tbl = {t: self.true_casacde_proba_func(self.g, self.p_dict, t, self.g_nx, using_log=True)
                     for t in possible_trees}
        log_pi_tbl = {t: tree_probability_gt(out_degree_dict, self.p_dict, t, using_log=True)
                      for t in possible_trees}

        log_p_T = np.array([log_p_tbl[t] for t in trees])
        log_pi_T = np.array([log_pi_tbl[t] for t in trees])

        sampling_weights = np.exp(log_p_T - log_pi_T)  # back to probabiliy

        weight_sum = sampling_weights.sum()
        if weight_sum > 0:
            sampling_weights /= weight_sum  # normlization
        else:
            # uniform sampling
            sampling_weights = np.ones(len(sampling_weights))
            sampling_weights /= sampling_weights.sum()

        # re-sampling trees by weights
        resampled_tree_idx = np.random.choice(self.n_samples,
                                              p=sampling_weights,
                                              replace=True,
                                              size=self.n_samples)

        resampled_trees = [trees[i] for i in resampled_tree_idx]
        return resampled_trees
