import sys
from graph_tool.all import load_graph

path = sys.argv[1]

g = load_graph('{}/graph.graphml'.format(path))
g.save('{}/graph.gt'.format(path))
