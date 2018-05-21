from graph_helpers import has_vertex


def check_tree_samples(qs, c, trees, every=1):
    # make sure the tree sampels are updated
    # exclude the last query
    for i, q in enumerate(qs[:-1]):
        if i % every == 0:
            for t in trees:
                if c[q] >= 0:
                    if isinstance(t, set):
                        assert q in t
                    else:
                        assert has_vertex(t, q)
                else:
                    if isinstance(t, set):
                        assert q not in t
                    else:
                        assert not has_vertex(t, q)


def check_error_esitmator(qs, c, est, every=1):
    # make sure the tree sampels are updated
    # exclude the last query
    for i, q in enumerate(qs[:-1]):
        if i % every == 0:
            if c[q] >= 0:
                # infected
                assert est._m[q, :].sum() == est.n_col
            else:
                # uninfected
                assert est._m[q, :].sum() == 0
    assert (est.n_row, est.n_col) == est._m.shape


def check_samples_so_far(g, sampler, estimator, obs_inf, obs_uninf):
    assert len(sampler.samples) == sampler.n_samples
    for v in obs_inf:
        for t in sampler.samples:
            assert isinstance(t, set), 'should be set'
            assert v in t, 'should be in sample'
            assert estimator._m[v, :].sum() == estimator.n_col

    assert estimator._m.shape == (g.num_vertices(), sampler.n_samples)
    for v in obs_uninf:
        for t in sampler.samples:
            assert isinstance(t, set), 'should be set'
            assert v not in t, 'should be in sample'
            assert estimator._m[v, :].sum() == 0
        

def check_probas_so_far(probas, inf, uninf):
    # print(inf)
    # print(uninf)
    for v in inf:
        assert probas[v] == 1.0
    for v in uninf:
        assert probas[v] == 0.0
