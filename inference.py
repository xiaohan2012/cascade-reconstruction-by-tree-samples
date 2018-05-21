from minimum_steiner_tree import min_steiner_tree


def infection_probability(g, obs, sampler, error_estimator):
    """
    infer infection probability over nodes given `obs` and using `sampler`
    """
    if error_estimator._m is None:
        error_estimator.build_matrix(sampler.samples)
    
    return error_estimator.unconditional_proba()
    
    # proba_values = np.array([node_occurrence_freq(n, sampler.samples)[0]
    #                          for n in remainig_nodes]) / sampler.n_samples
    # inf_probas = np.zeros(n_nodes)
    # inf_probas[remainig_nodes] = proba_values
    # return inf_probas


##################
## DEPRECATED
## Ignore it
##################
def infer_infected_nodes(g, obs, estimator=None, use_proba=True,
                         method="min_steiner_tree", min_inf_proba=0.5):
    """besides observed infections, infer other infected nodes
    if method is 'sampling', refer to infection_probability,

    `min_inf_proba` is the minimum infection probability to be considered "'infected'
    """
    assert method in {"min_steiner_tree", "sampling"}
    if method == 'min_steiner_tree':
        st = min_steiner_tree(g, obs)
        remain_infs = set(map(int, st.vertices()))
        return remain_infs
    else:  # sampling
        assert estimator is not None, 'sampling approach requires an estimator'
        inf_probas = estimator.unconditional_proba()
        if use_proba:
            return inf_probas
        else:
            return (inf_probas >= min_inf_proba).nonzero()[0]
