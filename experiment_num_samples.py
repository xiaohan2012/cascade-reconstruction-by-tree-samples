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

n_samples = [1250, 2500, 5000]
cascade_models = ['si']


# a batch of settings to iterate through
settings = [
    # for grqc
    {'graphs': ['grqc'],
     'obs_fractions': ["0.5"],
     'cascade_fractions': ["0.1"]},

    # for lattice and infectious
    {'graphs': ['infectious', 'lattice-1024'],
     'obs_fractions': ["0.5"],
     'cascade_fractions': ["0.2"]}
]

for setting in settings:
    graphs, obs_fractions, cascade_fractions = setting['graphs'], \
                                               setting['obs_fractions'], \
                                               setting['cascade_fractions']
    for graph, n_sample, cascade_model, obs_fraction, cascade_fraction \
            in product(
                graphs, n_samples, cascade_models, obs_fractions, cascade_fractions):
        if cascade_model == 'ic':
            # use reversed graph
            suffix = "uniform"
            graph_path = 'data/{}/graph_weighted_{}.gt'.format(graph, suffix + '_rev')
        else:
            suffix = "0.1"
            graph_path = 'data/{}/graph_weighted_{}.gt'.format(graph, suffix)

        print('reading graph from ', graph_path)
        g = load_graph(graph_path)
        edge_weights = g.edge_properties['weights']

        dataset_id = "{}-m{}-s{}-o{}-omuniform".format(graph, cascade_model, cascade_fraction, obs_fraction)
        print('method', method)
        print('dataset_id', dataset_id)

        input_dir = 'cascade/{}/'.format(dataset_id)
        output_dir = 'output/{}-n-samples-{}/{}/'.format(method, n_sample, dataset_id)
        eval_result_path = 'eval/{}-n-samples-{}/{}.pkl'.format(method, n_sample, dataset_id)

        if not os.path.exists(os.path.dirname(eval_result_path)):
            os.makedirs(os.path.dirname(eval_result_path))
                
        rows = Parallel(n_jobs=-1)(delayed(one_run)(g, edge_weights, input_path, output_dir, method,
                                                    n_sample=n_sample)
                                   for input_path in tqdm(glob(input_dir + '*.pkl')))
        assert len(rows) > 0, 'nothing calculated'

        scores = eval_map(input_dir, output_dir)

        summ = pd.Series(scores).describe()
        summ.to_pickle(eval_result_path)

        print('inf_result saved to', output_dir)
        print('evaluation saved to', eval_result_path)
        print(summ)
