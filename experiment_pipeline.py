# coding: utf-8

import os
import pandas as pd

from graph_tool import load_graph
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product

from eval_helpers import eval_map
from experiment import one_run
from helpers import is_processed, makedir_if_not_there

n_jobs = 4
n_sample = 1000

methods = ['our', 'pagerank', 'min-steiner-tree']

cascade_models = ['si', 'ic']

graphs = ['fb-messages', 'email-univ', 'infectious', 'lattice-1024', 'grqc']

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
    for graph, cascade_model, obs_fraction, cascade_fraction, method \
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
        print('method', method)
        print('dataset_id', dataset_id)

        input_dir = 'cascade/{}/'.format(dataset_id)
        output_dir = 'output/{}/{}/'.format(method, dataset_id)
        eval_result_path = 'eval/{}/{}.pkl'.format(method, dataset_id)

        makedir_if_not_there(output_dir)
        makedir_if_not_there(os.path.dirname(eval_result_path))

        rows = Parallel(n_jobs=n_jobs)(

            delayed(one_run)(
                g, edge_weights, input_path, output_dir, method,
                n_sample=n_sample)

            for input_path in tqdm(glob(input_dir + '*.pkl'))
            if not is_processed(input_path, output_dir))

        # assert len(rows) > 0, 'nothing calculated'

        scores = eval_map(input_dir, output_dir)

        summ = pd.Series(scores).describe()
        print(summ)
        summ.to_pickle(eval_result_path)
