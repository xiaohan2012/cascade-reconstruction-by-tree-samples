import numpy as np
from graph_helpers import pagerank_scores
from graph_tool.topology import shortest_distance
from tqdm import tqdm


def build_root_sampler_by_pagerank_score(g, obs, c, eps=0.0):
    # print('DEBUG: build_root_sampler_by_pagerank_score: eps={}'.format(eps))
    pr_score = pagerank_scores(g, obs, eps)
    # print(g)
    # print('len(obs): ', len(obs))
    # print(pr_score)
    nodes = np.arange(len(pr_score))  # shapes should be consistent

    def aux():
        return np.random.choice(nodes, size=1, p=pr_score)[0]

    return aux


def build_true_root_sampler(c):
    source = np.nonzero(c == 0)[0][0]

    def aux():
        return source

    return aux


def build_min_dist_sampler(g, obs, log=False):
    """minimum distance root sampler
    """
    # print('min_dist: select root')
    obs_set = set(obs)
    min_dist = 9999999999
    best_root = None
    iters = g.vertices()
    if log:
        iters = tqdm(iters, total=g.num_vertices())

    for v in iters:
        v = int(v)
        if v not in obs_set:
            dist = shortest_distance(
                g,
                source=v,
                target=obs,
                pred_map=False)
            if dist.sum() < min_dist:
                min_dist = dist.sum()
                best_root = v
    assert best_root is not None
    
    def aux():
        return best_root

    return aux


def build_out_degree_root_sampler(g, power=2):
    out_deg = np.power(g.degree_property_map('out').a, 2)
    out_deg_norm = out_deg / out_deg.sum()

    def aux():
        return np.random.choice(g.num_vertices(), p=out_deg_norm)
    return aux


def get_value_or_raise(d, key):
    if key in d:
        v = d[key]
        del d[key]
        return v
    else:
        raise KeyError('`{}` is not there'.format(d))


def get_root_sampler_by_name(name, **kwargs):
    if name == 'true_root':
        c = get_value_or_raise(kwargs, 'c')
        assert c is not None, 'cascade `c` should be give'
        return build_true_root_sampler(c)
    elif name == 'pagerank':
        g = get_value_or_raise(kwargs, 'g')
        obs = get_value_or_raise(kwargs, 'obs')
        c = get_value_or_raise(kwargs, 'c')
        return build_root_sampler_by_pagerank_score(g, obs, c, **kwargs)
    elif name is None:
        return None
    elif name == 'min_dist':
        g = get_value_or_raise(kwargs, 'g')
        obs = get_value_or_raise(kwargs, 'obs')
        return build_min_dist_sampler(g, obs)
    else:
        raise ValueError('valid name ', name)
