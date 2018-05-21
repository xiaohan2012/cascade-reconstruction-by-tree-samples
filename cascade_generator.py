import random
import math
import numpy as np
from copy import copy

from graph_tool import Graph, GraphView, PropertyMap
from graph_tool.topology import shortest_distance, label_components
from graph_tool.search import bfs_search

from tqdm import tqdm

from helpers import infected_nodes, timeout, sampling_weights_by_order
from graph_helpers import filter_graph_by_edges, get_leaves, BFSNodeCollector, reverse_bfs

MAXINT = np.iinfo(np.int32).max


def observe_cascade(c, source, q, method='uniform',
                    tree=None, source_includable=False):
    """
    given a cascade `c` and `source`,
    return a list of observed nodes according to probability `q`

    """
    all_infection = np.nonzero(c != -1)[0]
    if not source_includable:
        all_infection = list(set(all_infection) - {source})
    num_obs = int(math.ceil(len(all_infection) * q))

    if num_obs < 2:
        num_obs = 2

    if method == 'uniform':
        return np.random.permutation(all_infection)[:num_obs]
    elif method == 'late':
        return np.argsort(c)[-num_obs:]
    elif method == 'leaves':
        assert tree is not None, 'to get the leaves, the cascade tree is required'
        # extract_steiner_tree(tree, )
        nodes_in_order = reverse_bfs(tree)
        return nodes_in_order[:num_obs]
    elif method == 'bfs-head':
        assert tree is not None, 'the cascade tree is required'
        vis = BFSNodeCollector()
        bfs_search(GraphView(tree, directed=False), source, vis)
        sampling_weights_by_order
        vis.nodes_in_order
        return vis.nodes_in_order[:num_obs]  # head
    elif method == 'bfs-tail':
        assert tree is not None, 'the cascade tree is required'
        vis = BFSNodeCollector()
        bfs_search(GraphView(tree, directed=False), source, vis)
        return vis.nodes_in_order[-num_obs:]  # tail
    else:
        raise ValueError('unknown method {}'.format(method))


@timeout(2)
def si(g, p, source=None, stop_fraction=0.5):
    """
    g: the graph
    p: edge-wise infection probability
    stop_fraction: stopping if more than N x stop_fraction nodes are infected
    """
    weighted = False
    if isinstance(p, PropertyMap):
        weighted = True
    else:
        # is float and uniform
        assert 0 < p and p <= 1
        
    if source is None:
        source = random.choice(np.arange(g.num_vertices()))
    infected = {source}
    infection_times = np.ones(g.num_vertices()) * -1
    infection_times[source] = 0
    time = 0
    edges = []

    stop = False

    infected_nodes_until_t = copy(infected)
    while True:
        infected_nodes_until_t = copy(infected)
        # print('current cascade size: {}'.format(len(infected_nodes_until_t)))
        time += 1
        for i in infected_nodes_until_t:
            vi = g.vertex(i)
            for e in vi.all_edges():
                if weighted:
                    inf_proba = p[e]
                else:
                    inf_proba = p
                vj = e.target()
                j = int(vj)
                rand = random.random()
                # print('rand=', rand)
                # print('inf_proba=', inf_proba)
                # print('{} infected?'.format(j), j not in infected)
                if j not in infected and rand <= inf_proba:
                    # print('SUCCESS')
                    infected.add(j)
                    infection_times[j] = time
                    edges.append((i, j))

                    # stop when enough nodes have been infected
                    if (len(infected) / g.num_vertices()) >= stop_fraction:
                        stop = True
                        break
            if stop:
                break
        if stop:
            break
        
    tree = Graph(directed=True)
    for _ in range(g.num_vertices()):
        tree.add_vertex()
    for u, v in edges:
        tree.add_edge(u, v)
    return source, infection_times, tree


def sample_graph_by_p(g, p):
    """
    for IC model
    graph_tool version of sampling a graph
    mask the edge according to probability p and return the masked graph

    g: the graph
    p: float or np.array
    """
    if isinstance(p, PropertyMap):
        p = p.a
    flags = (np.random.random(p.shape) <= p)
    p = g.new_edge_property('bool')
    p.set_2d_array(flags)
    return GraphView(g, efilt=p)


def get_infection_time(g, source, return_edges=False):
    """for IC model
    """
    time, pred_map = shortest_distance(g, source=source, pred_map=True)
    time = np.array(time.a)
    time[time == MAXINT] = -1
    if return_edges:
        edges = []
        reached = infected_nodes(time)
        for v in reached:
            # print(v)
            if pred_map[v] >= 0 and pred_map[v] != v:
                edges.append((pred_map[v], v))
        return time, edges
    else:
        return time


def ic(g, p, source=None, return_tree_edges=False,
       min_size=0, max_size=1e10):
    """
    graph_tool version of simulating cascade
    return np.ndarray on vertices as the infection time in cascade
    uninfected node has dist -1
    """
    if source is None:
        source = random.choice(np.arange(g.num_vertices(), dtype=int))
    gv = sample_graph_by_p(g, p)

    times = get_infection_time(gv, source, return_edges=False)
    size = len(infected_nodes(times))

    if size < min_size or size > max_size:
        # size does not fit
        # early stopping to save time
        return source, times, None
    
    stuff = get_infection_time(gv, source, return_edges=return_tree_edges)

    if not return_tree_edges:
        times = stuff
        tree_edges = None
    else:
        times, tree_edges = stuff
        # tree = filter_graph_by_edges(gv, tree_edges)
    
    return source, times, tree_edges


def incremental_simulation(g, c, p, num_nodes, return_new_edges=False):
    """incrementally add edges to given cascade
    num_nodes is passed bacause vfilt might be passed
    """
    # print('incremental_simulation -> g', g)
    gv = sample_graph_by_p(g, p)

    new_infected_nodes = set(infected_nodes(c))
    comp = label_components(gv)[0]
    covered_cids = set()
    for v in infected_nodes(c):
        cid = comp[v]
        if cid not in covered_cids:
            new_infected_nodes |= set((comp.a == cid).nonzero()[0])
            covered_cids.add(cid)
    
    new_c = np.ones(g.num_vertices()) * (-1)
    new_c[list(new_infected_nodes)] = 1

    if return_new_edges:
        raise Exception("`return_new_edges` not supported anymore")
    else:
        return new_c

