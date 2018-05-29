import os
import pandas as pd

from graph_tool import load_graph
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product

from eval_helpers import eval_map
from experiment import one_run
# infection_proba = 0.1

method = 'our'

cascade_model = 'si'
root_sampler_names = ['true_root']
if cascade_model == 'si':
    suffix = "0.1"
elif cascade_model == 'ic':
    suffix = "uniform"


# a batch of settings to iterate through
settings = [
    # for grqc
    {'graphs': ['grqc'],
     'obs_fractions': ["0.5"],
     'cascade_fractions': ["0.05", "0.1", "0.15", "0.2", "0.25"]},
    {'graphs': ['grqc'],
     'obs_fractions': ["0.5", "0.6", "0.7", "0.8", "0.9"],
     'cascade_fractions': ["0.05"]},

    # for lattice and infectious
    {'graphs': ['infectious', 'lattice-1024'],
     'obs_fractions': ["0.5"],
     'cascade_fractions': ["0.1", "0.2", "0.3", "0.4", "0.5"]},
    {'graphs': ['infectious', 'lattice-1024'],
     'obs_fractions': ["0.5", "0.6", "0.7", "0.8", "0.9"],
     'cascade_fractions': ["0.1"]}
]

for setting in settings:
    graphs, obs_fractions, cascade_fractions = setting['graphs'], \
                                               setting['obs_fractions'], \
                                               setting['cascade_fractions']
    for graph, obs_fraction, cascade_fraction, root_sampler_name \
            in product(
                graphs, obs_fractions, cascade_fractions, root_sampler_names):
        if cascade_model == 'ic':
            # use reversed graph
            graph_path = 'data/{}/graph_weighted_{}.gt'.format(graph, suffix + '_rev')
        else:
            graph_path = 'data/{}/graph_weighted_{}.gt'.format(graph, suffix)

        print('reading graph from ', graph_path)
        g = load_graph(graph_path)
        edge_weights = g.edge_properties['weights']

        dataset_id = "{}-m{}-s{}-o{}-omuniform".format(graph, cascade_model, cascade_fraction, obs_fraction)
        print('method', method)
        print('dataset_id', dataset_id)

        input_dir = 'cascade/{}/'.format(dataset_id)
        output_dir = 'output/{}-{}/{}/'.format(method, root_sampler_name, dataset_id)
        eval_result_path = 'eval/{}-{}/{}.pkl'.format(method, root_sampler_name, dataset_id)

        if not os.path.exists(os.path.dirname(eval_result_path)):
            os.makedirs(os.path.dirname(eval_result_path))
                
        rows = Parallel(n_jobs=-1)(delayed(one_run)(g, edge_weights, input_path, output_dir, method,
                                                    root_sampler_name=root_sampler_name)
                                   for input_path in tqdm(glob(input_dir + '*.pkl')))
        assert len(rows) > 0, 'nothing calculated'

        scores = eval_map(input_dir, output_dir)

        summ = pd.Series(scores).describe()
        summ.to_pickle(eval_result_path)

        print('inf_result saved to', output_dir)
        print('evaluation saved to', eval_result_path)
        print(summ)
        
