import numpy as np

EPS = 1e-15
# EPS = 0.0


class TreeBasedStatistics:
    def __init__(self, g, trees=None):
        self._g = g
        self.n_row = g.num_vertices()
        self.n_col = None
        self._m = None

        if trees is not None:
            self.build_matrix(trees)

    def build_matrix(self, trees):
        """trees: list of set of ints
        """
        self.n_row = self._g.num_vertices()
        self.n_col = len(trees)
        self._m = np.zeros((self.n_row, self.n_col), dtype=np.bool)
        for i, t in enumerate(trees):
            for v in t:
                self._m[v, i] = True

    def update_trees(self, trees, node_info):
        invalid_tree_indices = set()
        for n, v in node_info.items():
            invalid_tree_indices |= set((self._m[n, :] != v).nonzero()[0])

        assert len(invalid_tree_indices) <= len(trees), \
            "need enough trees to update ({} vs {})".format(len(invalid_tree_indices), len(trees))
        # print('invalid_tree_indices', invalid_tree_indices)
        for i, t in zip(invalid_tree_indices, trees):
            self._m[:, i] = False
            for v in t:
                self._m[v, i] = True

    def count(self, query, condition, targets, return_denum=False):
        """
        count node occurrence frequency in trees that satisfy condition on `query` and `condition`

        return:
        1. an array |targets|
        2. optionally, |{tree that satisfy tree[query]==condition}| is returned if `return_denum` is True

        """
        mask = (self._m[query, :] == condition).nonzero()[0]
        # print('mask', mask)
        # print('np.asarray(targets)[:, None]', np.array(list(targets))[:, None])
        try:
            sub_m = self._m[np.asarray(list(targets))[:, None], mask]
        except IndexError as exc:
            raise IndexError("targets have value: {}".format(list(targets))) from exc

        if not return_denum:
            return sub_m.sum(axis=1)
        else:
            return sub_m.sum(axis=1), len(mask)

    def unconditional_count(self, targets=None):
        assert self._m is not None, 'occurence matrix not initialized yet'
        if targets is None:
            sub_m = self._m
        else:
            sub_m = self._m[np.asarray(list(targets)), :]
        return sub_m.sum(axis=1)

    def unconditional_proba(self, targets=None):
        return self.unconditional_count(targets) / self.n_col

    def filter_out_extreme_targets(self, targets=None, min_value=0):
        """return targets whose min(p, 1-p) > min_value,
        where p is the unconditional probability
        """
        if targets is None:
            targets = np.arange(self.n_row)
        proba1 = self.unconditional_proba(targets)
        proba0 = 1 - proba1
        min_proba = np.minimum(proba0, proba1)
        indices = np.nonzero(min_proba > min_value)[0]
        return np.array(list(targets))[indices]

    def proba(self, *args, **kwargs):
        num, denum = self.count(*args, **kwargs, return_denum=True)
        return num / denum

    def _smooth_extreme_vals(self, v):
        # remove zero and one
        v[(v == 0) | (v == 1)] = EPS
        return v

    def _sum_entropy(self, p, weights):
        ents_arr = -(p * np.log(p) + (1-p) * np.log(1-p))
        return (ents_arr * weights).sum()

    def prediction_error(self, query, condition, targets):
        p = self.proba(query, condition, targets)
        p = self._smooth_extreme_vals(p)
        return self._sum_entropy(p, np.ones(len(targets)))

    def query_score(self, query, targets, node_weights=None, return_verbose=False):
        num0, denum0 = self.count(query, 0, targets, return_denum=True)
        num1, denum1 = self.count(query, 1, targets, return_denum=True)

        assert len(targets) == len(num0)

        p0, p1 = (self._smooth_extreme_vals(num0 / denum0),
                  self._smooth_extreme_vals(num1 / denum1))

        if node_weights is None:
            node_weights = np.ones(p0.shape)  # equal weight
        elif node_weights == 'uncond_proba':
            # can be cached for each query selection
            node_weights = self.unconditional_proba(targets)

        assert node_weights.shape == p0.shape, 'shape unmatch: {}, {}'.format(
            node_weights.shape,
            p0.shape)

        weights = np.array([denum0, denum1]) / self.n_col
        errors = np.array([self._sum_entropy(p0, node_weights), self._sum_entropy(p1, node_weights)])

        if False:
            print("query: ", query)
            print("p(uninfected)={}, p(infected)={}".format(weights[0], weights[1]))
            print('p(infected | q uninfected)={}'.format(p0))
            print('p(infected | q infected)={}'.format(p1))
            print('errors={}'.format(errors))

        if not return_verbose:
            return (weights * errors).sum()
        else:
            return (weights * errors).sum(), {
                'weights': (weights[0], weights[1]),
                'p0': p0,
                'p1': p1,
                'errors': errors
            }
