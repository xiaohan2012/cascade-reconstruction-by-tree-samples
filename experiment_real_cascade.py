# coding: utf-8

import os
import numpy as np
import pandas as pd

from graph_tool import load_graph
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product

from eval_helpers import eval_map
from experiment import one_run
from helpers import makedir_if_not_there, is_processed
from graph_tool import openmp_set_num_threads

openmp_set_num_threads(1)

parallel = True
max_n_jobs = -1
n_sample = 1000

# , 'pagerank', 'min-steiner-tree'
methods = ['our']

root_sampler = 'pagerank'

infection_proba = 0.1

# a batch of settings to iterate through
settings = [
    {'graphs': ['digg'],
     'obs_fractions': np.linspace(0.6, 0.9, 4)}
]

for setting in settings:
    graphs, obs_fractions = setting['graphs'], \
                            setting['obs_fractions'],
    for graph, obs_fraction, method \
            in product(
                graphs, obs_fractions, methods
            ):
        g = load_graph('data/{}/graph_weighted_{}.gt'.format(graph, infection_proba))

        # assume the infection probability is fixed
        edge_weights = g.new_edge_property('float')
        edge_weights.a = infection_proba

        dataset_id = "{}-o{:.1f}-omuniform".format(graph, obs_fraction)
        print('method', method)
        print('dataset_id', dataset_id)

        input_dir = 'cascade/{}/'.format(dataset_id)
        if method != 'our':
            output_dir = 'output/{}/{}/'.format(method, dataset_id)
            eval_result_path = 'eval/{}/{}.pkl'.format(method, dataset_id)
        else:
            output_dir = 'output/{}-{}/{}/'.format(method, root_sampler, dataset_id)
            eval_result_path = 'eval/{}-{}/{}.pkl'.format(method, root_sampler, dataset_id)

        eval_dir = os.path.dirname(eval_result_path)
        print('eval_dir', eval_dir)
        makedir_if_not_there(eval_dir)
        makedir_if_not_there(output_dir)

        if parallel:
            print('parallel: ON')
            if method == 'min-steiner-tree':
                n_jobs = 4  # memory reason
            else:
                n_jobs = max_n_jobs
            print('n_jobs', n_jobs)
            rows = Parallel(n_jobs=n_jobs)(delayed(one_run)(
                g, edge_weights, input_path, output_dir, method,
                root_sampler_name=root_sampler,
                n_sample=n_sample)
                                           for input_path in tqdm(glob(input_dir + '*.pkl'))
                                           if not is_processed(input_path, output_dir))
        else:
            print('parallel: OFF')
            for input_path in tqdm(glob(input_dir + '*.pkl')):
                one_run(g, edge_weights, input_path, output_dir, method,
                        root_sampler_name=root_sampler,
                        n_sample=n_sample)
            
        # assert len(rows) > 0, 'nothing calculated'

        scores = eval_map(input_dir, output_dir)

        summ = pd.Series(scores).describe()
        print(summ)
        summ.to_pickle(eval_result_path)
