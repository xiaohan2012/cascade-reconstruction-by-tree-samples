import time
import numpy as np
from graph_tool.generation import lattice

from core import sample_steiner_trees
from random_steiner_tree import util
from tqdm import tqdm

N = 10

def wrapper_for_sampling_procedure(g, obs, method, gi,
                                   return_tree_nodes):
    ts = time.time()
    sample_steiner_trees(g, obs,
                         method,
                         n_samples=1,
                         gi=gi,
                         return_tree_nodes=return_tree_nodes)
    te = time.time()
    return te - ts


def run(g, N, method, return_tree_nodes):
    t_sum = 0
    gi = util.from_gt(g, None)
    for i in tqdm(range(N), total=N):
        obs = np.random.choice(np.arange(g.num_vertices()), 10, replace=False)
        t_sum += wrapper_for_sampling_procedure(g, obs, method, gi,
                                                return_tree_nodes)
    print('{} took {:.2f} sec on average'.format(method, t_sum / N))
        

# return_tree_nodes plays a big difference here
# construting the GraphView is very time consuming

g_dim = 100
return_tree_nodes = True
g = lattice((g_dim, g_dim))

run(g, N, 'cut_naive', return_tree_nodes)
run(g, N, 'cut', return_tree_nodes)
run(g, N, 'loop_erased', return_tree_nodes)
