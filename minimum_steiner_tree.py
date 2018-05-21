import itertools
import numpy as np
from graph_tool import Graph, GraphView
from graph_tool.search import bfs_search
from graph_tool.topology import min_spanning_tree

from graph_helpers import init_visitor, extract_edges_from_pred


def build_closure(g, terminals,
                  debug=False,
                  verbose=False):
    """build the transitive closure on terminals"""
    def get_edges(dist, root, terminals):
        """get adjacent edges to root with weight"""
        return ((root, t, dist[t])
                for t in terminals
                if dist[t] != -1 and t != root)

    terminals = list(terminals)
    gc = Graph(directed=False)

    gc.add_vertex(g.num_vertices())

    edges_with_weight = set()
    r2pred = {}  # root to predecessor map (from bfs)
    
    # bfs to all other nodes
    for r in terminals:
        if debug:
            print('root {}'.format(r))
        vis = init_visitor(g, r)
        bfs_search(g, source=r, visitor=vis)
        new_edges = set(get_edges(vis.dist, r, terminals))
        if debug:
            print('new edges {}'.format(new_edges))
        edges_with_weight |= new_edges
        r2pred[r] = vis.pred

    for u, v, c in edges_with_weight:
        gc.add_edge(u, v)

    # edge weights
    eweight = gc.new_edge_property('int')
    weights = np.array([c for _, _, c in edges_with_weight])
    eweight.set_2d_array(weights)

    # 
    vfilt = gc.new_vertex_property('bool')
    vfilt.a = False
    for v in terminals:
        vfilt[v] = True
    gc.set_vertex_filter(vfilt)
    return gc, eweight, r2pred


def min_steiner_tree(g, obs_nodes, debug=False, verbose=False):
    if g.num_vertices() == len(obs_nodes):
        print('it\'s a minimum spanning tree problem')
        
    gc, eweight, r2pred = build_closure(g, obs_nodes,
                                        debug=debug, verbose=verbose)
    # print('gc', gc)

    tree_map = min_spanning_tree(gc, eweight, root=None)
    tree = GraphView(gc, directed=False, efilt=tree_map)

    tree_edges = set()
    # print('tree', tree)
    for e in tree.edges():
        u, v = map(int, e)
        recovered_edges = extract_edges_from_pred(u, v, r2pred[u])
        assert recovered_edges, 'empty!'
        for i, j in recovered_edges:
            tree_edges.add(((i, j)))
            
    tree_nodes = list(set(itertools.chain(*tree_edges)))

    vfilt = g.new_vertex_property('bool')
    vfilt.set_value(False)
    for n in tree_nodes:
        vfilt[n] = True
    
    efilt = g.new_edge_property('bool')
    for i, j in tree_edges:
        efilt[g.edge(i, j)] = 1
    return GraphView(g, efilt=efilt, vfilt=vfilt)
