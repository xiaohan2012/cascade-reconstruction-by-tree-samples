import  matplotlib as mpl
mpl.use('Agg')

import sys
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

from eval_helpers import top_k_infection_precision_recall
from graph_helpers import load_graph_by_name, gen_random_spanning_tree

from inference import infection_probability
from experiment import gen_input


graph_name = sys.argv[1]
g = load_graph_by_name(graph_name)

n_rounds = 100
n_samples = 200
subset_size = 50

if graph_name == 'lattice':
    stop_fraction = 0.25
    obs_fraction = 0.25
    ks = np.arange(1, 30, 2)
elif graph_name == 'karate':
    stop_fraction = 0.5
    obs_fraction = 0.25
    ks = np.arange(1, 18, 2)
elif graph_name == 'dolphin':
    stop_fraction = 0.5
    obs_fraction = 0.25
    ks = np.arange(1, 30, 2)
    
sp_trees = [gen_random_spanning_tree(g) for _ in range(n_samples)]


def one_batch_for_method(n_rounds, inference_kwargs, ks=[5, 10, 15, 20]):
    scores = {k: [] for k in ks}
    for i in tqdm(range(n_rounds)):
        obs, c = gen_input(g, stop_fraction=stop_fraction, q=obs_fraction, p=1)
        probas = infection_probability(g, obs, **inference_kwargs)
        for k in ks:
            prec, rec = top_k_infection_precision_recall(g, probas, c, obs, k=k)
            scores[k].append((prec, rec))
    return scores

sampling_scores = one_batch_for_method(
    n_rounds, inference_kwargs={'method': 'sampling', 'sp_trees': sp_trees},
    ks=ks)
subset_scores = one_batch_for_method(
    n_rounds, inference_kwargs={'method': 'sampling', 'sp_trees': sp_trees, 'subset_size': subset_size},
    ks=ks)
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

for scores in [sampling_scores, subset_scores]:
    # for each method
    mean_scores = np.asarray([np.mean(scores[k], axis=0)
                              for k in ks])
    # precision
    ax[0].plot(ks, mean_scores[:, 0], '-o')
    
    # recall
    ax[1].plot(ks, mean_scores[:, 1], '-o')
ax[0].set_xlabel('k')
ax[1].set_xlabel('k')

ax[0].set_title('precision')
ax[1].set_title('recall')
plt.legend(['full', 'subset'], loc='lower right')
fig.savefig('figs/inference_method_comparison_{}.pdf'.format(graph_name))
