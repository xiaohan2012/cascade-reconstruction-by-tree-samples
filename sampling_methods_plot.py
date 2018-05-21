# coding: utf-8

import matplotlib as mpl
mpl.use('pdf')

import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

from graph_helpers import load_graph_by_name


graph_name = 'grqc'
suffix = 's0.03'
aspects = ['roc', 'ap', 'precision', 'recall', 'f1']

qs = ['0.1', '0.25', '0.5', '0.75']

eval_metric = 'mean'

for aspect in aspects:
    root_sampling_method = 'random_root'

    g = load_graph_by_name(graph_name, weighted=True, suffix='_'+suffix)

    methods = ['pagerank-eps0.0', 'pagerank-eps0.5', 'pagerank-eps1.0', 'random_root', 'true root']

    columns_to_plot = []
    for q in qs:
        result_path = 'eval_result/{}-{}-q{}-by_root_sampling_methods.pkl'.format(graph_name, suffix, q)
        row = pkl.load(open(result_path, 'rb'))
        print('q={}'.format(q))
        print('-' * 10)
        print(row[aspect][root_sampling_method])
        columns_to_plot.append(row[aspect][root_sampling_method].loc['50%'].as_matrix())

    rows_to_plot = np.array(columns_to_plot).T
    xs = list(map(float, qs))

    plt.clf()
    for r in rows_to_plot:
        plt.plot(xs, r, 'o-')
    plt.legend(row[aspect][root_sampling_method].columns, loc='best')
    plt.xlabel('obs fraction')
    plt.ylabel(eval_metric + ' ' + aspect)

    output_path = 'figs/sampling-methods-eval-by-q/{}-{}-{}.pdf'.format(graph_name, suffix, aspect)
    print('output to {}'.format(output_path))
    plt.savefig(output_path)
