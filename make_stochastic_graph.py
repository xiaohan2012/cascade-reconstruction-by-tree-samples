# coding: utf-8

import sys
import numpy as np
from graph_helpers import load_graph_by_name
from preprocess_graph import reverse_edge_weights


graph_name = sys.argv[1]
g = load_graph_by_name(graph_name, weighted=True)


w = g.new_edge_property('float')
in_deg = g.degree_property_map('in', weight=None)
for u in g.vertices():
    for v in g.vertex(u).in_neighbours():  # v -> u
        w[g.edge(v, u)] = 1 / in_deg[u]


in_deg_weighted = g.degree_property_map('in', weight=w)
assert np.all(np.isclose(in_deg_weighted.a, 1)), 'maybe self-loops are not removed'


g.edge_properties['weights'] = w

g.save('data/{}/graph_sto.gt'.format(graph_name))

rev_g = reverse_edge_weights(g)

out_deg_weighted = g.degree_property_map('out', weight=rev_g.edge_properties['weights'])
assert np.all(np.isclose(out_deg_weighted.a, 1))


rev_g.save('data/{}/graph_sto_rev.gt'.format(graph_name))

