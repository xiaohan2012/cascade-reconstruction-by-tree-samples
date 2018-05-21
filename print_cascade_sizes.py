import pickle as pkl
import pandas as pd
import numpy as np
from glob import glob
from graph_helpers import load_graph_by_name
from helpers import infected_nodes, cascade_source
from collections import Counter

graph = 'grqc-sto'
model = 'ic'
# suffix = '_tmp'
# cascade_fraction = 0
suffix = ''
cascade_fraction = 0.01
obs_frac = "0.2"
cascade_dir = 'cascade-weighted'

dirname = '{}/{}-m{}-s{}-o{}/*'.format(
    cascade_dir,
    graph, model, cascade_fraction, obs_frac)

g = load_graph_by_name(graph, weighted=True, suffix=suffix)

gprop = g.graph_properties
if 'p_min' in gprop:
    p_min, p_max = gprop['p_min'], gprop['p_max']
    print('p_min={}, p_max={}'.format(p_min, p_max))
else:
    print('external weight initialization')

os = [pkl.load(open(p, 'rb'))[0] for p in glob(dirname)]
cs = [pkl.load(open(p, 'rb'))[1] for p in glob(dirname)]
obs_sizes = [len(o) for o in os]
c_sizes = [len(infected_nodes(c)) for c in cs]
roots = list(map(cascade_source, cs))
print('roots freq:')
print(Counter(roots).most_common(10))

obs_cnt = Counter([tuple(sorted(o)) for o in os])
print('top cascade freq:')
for _, c in obs_cnt.most_common(10):
    print('freq:', c)

print('cascade size describe:')
print(pd.Series(c_sizes).describe())
print('-' * 10)
print('fraction', np.mean(c_sizes) / g.num_vertices())

print('-' * 10)
print('obs_sizes describe:')
print(pd.Series(obs_sizes).describe())

