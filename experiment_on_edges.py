# coding: utf-8

import os
import pandas as pd

from graph_tool import load_graph
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product

from eval_helpers import eval_edge_map
from experiment import one_run_for_edge
from helpers import is_processed, makedir_if_not_there

n_jobs = 8
n_sample = 1000

# method name and root_sampler
methods = [('our', None),
           ('our', 'true-root'),
           ('our', 'min_dist'),
           ('min-steiner-tree', None)]

cascade_models = ['ic', 'si']

graphs = ['lattice-1024', 'infectious', 'fb-messages', 'email-univ', 'grqc']
# graphs = ['grqc']

# a batch of settings to iterate through
settings = [
    {'graphs': graphs,
     'obs_fractions': ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
     'cascade_fractions': ["0.1"]},
    {'graphs': graphs,
     'obs_fractions': ["0.5"],
     'cascade_fractions': ["0.1", "0.2", "0.3", "0.4", "0.5"]}
]

for setting in settings:
    graphs, obs_fractions, cascade_fractions = setting['graphs'], \
                                               setting['obs_fractions'], \
                                               setting['cascade_fractions']
    for graph, cascade_model, obs_fraction, cascade_fraction, (method, root_sampler) \
            in product(
                graphs, cascade_models, obs_fractions, cascade_fractions, methods
            ):

        if cascade_model == 'ic':
            # use reversed graph
            suffix = "uniform"
            graph_path = 'data/{}/graph_weighted_{}.gt'.format(graph, suffix + '_rev')
        else:
            suffix = "0.1"
            graph_path = 'data/{}/graph_weighted_{}.gt'.format(graph, suffix)

        g = load_graph(graph_path)
        edge_weights = g.edge_properties['weights']

        dataset_id = "{}-m{}-s{}-o{}-omuniform".format(graph, cascade_model, cascade_fraction, obs_fraction)
        # print('method', method)
        # print('dataset_id', dataset_id)

        input_dir = 'cascade-with-edges/{}/'.format(dataset_id)

        if method == 'our' and root_sampler is not None:
            output_dir = 'output-edges/{}-{}/{}/'.format(method, root_sampler, dataset_id)
            eval_result_path = 'eval-edges/{}-{}/{}.pkl'.format(method, root_sampler, dataset_id)
        else:
            output_dir = 'output-edges/{}/{}/'.format(method, dataset_id)
            eval_result_path = 'eval-edges/{}/{}.pkl'.format(method, dataset_id)

        eval_dir = os.path.dirname(eval_result_path)
        print('output_dir', output_dir)
        print('eval_dir', eval_dir)

        makedir_if_not_there(output_dir)
        makedir_if_not_there(eval_dir)

        rows = Parallel(n_jobs=n_jobs)(

            delayed(one_run_for_edge)(
                g, edge_weights, input_path, output_dir, method,
                root_sampler=root_sampler,
                n_sample=n_sample)

            for input_path in tqdm(glob(input_dir + '*.pkl'))
            if not is_processed(input_path, output_dir))

        # assert len(rows) > 0, 'nothing calculated'

        if not os.path.exists(eval_result_path):
            scores = eval_edge_map(g, input_dir, output_dir)

            summ = pd.Series(scores).describe()
            print(summ)
            summ.to_pickle(eval_result_path)
        else:
            print('evaluated already')
