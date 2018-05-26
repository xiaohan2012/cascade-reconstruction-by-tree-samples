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

infection_proba = 0.1

graphs = ['infectious', 'lattice-1024']
methods = ['min-steiner-tree']
# 'our',
# graphs = ['grqc']
# obs_fractions = ["0.5"]
obs_fractions = ["0.5", "0.6", "0.7", "0.8", "0.9"]
# cascade_fractions = ["0.1", "0.2", "0.3", "0.4", "0.5"]
cascade_fractions = ["0.1"]
# cascade_fractions = ["0.05"]
# cascade_fractions = ["0.1", "0.15", "0.2", "0.25", "0.05"]

for graph, obs_fraction, cascade_fraction, method \
        in product(
            graphs, obs_fractions, cascade_fractions, methods
        ):
    g = load_graph('data/{}/graph_weighted_{}.gt'.format(graph, infection_proba))
    edge_weights = g.edge_properties['weights']

    dataset_id = "{}-msi-s{}-o{}-omuniform".format(graph, cascade_fraction, obs_fraction)
    print('method', method)
    print('dataset_id', dataset_id)

    input_dir = 'cascade/{}/'.format(dataset_id)
    output_dir = 'output/{}/{}/'.format(method, dataset_id)
    eval_result_path = 'eval/{}/{}.pkl'.format(method, dataset_id)

    if not os.path.exists(os.path.dirname(eval_result_path)):
        os.makedirs(os.path.dirname(eval_result_path))

    rows = Parallel(n_jobs=-1)(delayed(one_run)(g, edge_weights, input_path, output_dir, method)
                               for input_path in tqdm(glob(input_dir + '*.pkl')))
    assert len(rows) > 0, 'nothing calculated'

    scores = eval_map(input_dir, output_dir)

    summ = pd.Series(scores).describe()
    print(summ)
    summ.to_pickle(eval_result_path)
