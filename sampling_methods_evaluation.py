# coding: utf-8


import pandas as pd
import numpy as np
import pickle as pkl
from graph_tool import openmp_set_num_threads
from tqdm import tqdm

from graph_helpers import load_graph_by_name
from helpers import infected_nodes

from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)


def accumulate_score(stuff, eval_func):
    scores_by_root_sampling_method = {}
    for root_sampling_method, data in stuff.items():
        scores_by_root_sampling_method[root_sampling_method] = []
        for row in tqdm(data):
            c, obs = row['c'], row['obs']
            inf_nodes = infected_nodes(c)
            y_true = np.zeros((len(c), ))
            y_true[inf_nodes] = 1
            mask = np.array([(i not in obs) for i in range(len(c))])

            score = {}
            # names = ['random', 'st_naive', 'st_inc']
            names = ['random', 'st_naive']
            random_inf_p = np.random.random(g.num_vertices())
            for name, inf_probas in zip(names, [random_inf_p,
                                                row['st_naive_probas']]):
                                                # row['st_tree_inc_probas']]):
                score[name] = eval_func(y_true[mask], inf_probas[mask])
            scores_by_root_sampling_method[root_sampling_method].append(score)
    return scores_by_root_sampling_method


def precision_score_half_threshold(y_true, y_pred):
    pred_labels = np.asarray((y_pred >= 0.5), dtype=np.bool)
    return precision_score(y_true, pred_labels)


def recall_score_half_threshold(y_true, y_pred):
    pred_labels = np.asarray((y_pred >= 0.5), dtype=np.bool)
    return recall_score(y_true, pred_labels)


def f1_score_half_threshold(y_true, y_pred):
    pred_labels = np.asarray((y_pred >= 0.5), dtype=np.bool)
    return f1_score(y_true, pred_labels)


def describe_stuff(scores_by_root_sampling_method):
    df_by_root_sampling_method = {}
    for name, recs in scores_by_root_sampling_method.items():
        df_by_root_sampling_method[name] = pd.DataFrame.from_records(recs).describe()
    return df_by_root_sampling_method


def print_result(df_by_keys):
    # 'pagerank-eps0.0', 'pagerank-eps0.5', 'pagerank-eps1.0',
    keys = ['random_root', 'true_root']
    for k in keys:
        df = df_by_keys[k]
        print(k)
        print('-' * 10)
        print(df)


if __name__ == '__main__':
    import argparse

    openmp_set_num_threads(1)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--graph', help='graph name')
    parser.add_argument('-f', '--graph_suffix',
                        required=True,
                        help='suffix of graph name')
    parser.add_argument('-i', '--input_path', help='graph name')    
    parser.add_argument('-q', '--obs_fraction',
                        type=float,
                        help='fraction of observed infections')
    parser.add_argument('-o', '--output_path',
                        help='output_path')

    args = parser.parse_args()

    print("Args:")
    print('-' * 10)
    for k, v in args._get_kwargs():
        print("{}={}".format(k, v))

    graph_name = args.graph
    suffix = args.graph_suffix
    output_path = args.output_path
    
    q = args.obs_fraction

    g = load_graph_by_name(graph_name, weighted=True, suffix=suffix)

    stuff = pkl.load(open(args.input_path, 'rb'))

    ap_scores_by_root_sampling_method = accumulate_score(stuff, average_precision_score)
    roc_scores_by_root_sampling_method = accumulate_score(stuff, roc_auc_score)

    precision_by_root_sampling_method = accumulate_score(stuff, precision_score_half_threshold)
    recall_by_root_sampling_method = accumulate_score(stuff, recall_score_half_threshold)

    f1_scores_by_root_sampling_method = accumulate_score(stuff, f1_score_half_threshold)

    ap_df_by_root_sampling_method = describe_stuff(ap_scores_by_root_sampling_method)
    roc_df_by_root_sampling_method = describe_stuff(roc_scores_by_root_sampling_method)
    precision_df_by_root_sampling_method = describe_stuff(precision_by_root_sampling_method)
    recall_df_by_root_sampling_method = describe_stuff(recall_by_root_sampling_method)
    f1_df_by_root_sampling_method = describe_stuff(f1_scores_by_root_sampling_method)

    # output_path = 'eval_result/{}-{}-q{}-by_root_sampling_methods.pkl'.format(graph_name, suffix, q)
    print('output_path', output_path)

    result = {
        'ap': ap_df_by_root_sampling_method,
        'roc': roc_df_by_root_sampling_method,
        'precision': precision_df_by_root_sampling_method,
        'recall': recall_df_by_root_sampling_method,
        'f1': f1_df_by_root_sampling_method
    }
    print('dump to {}'.format(output_path))
    pkl.dump(result, open(output_path, 'wb'))

    print('ap')
    print_result(ap_df_by_root_sampling_method)

    print('roc')
    print_result(roc_df_by_root_sampling_method)

    print('precision:')
    print_result(precision_df_by_root_sampling_method)

    print('recall')
    print_result(recall_df_by_root_sampling_method)

    print('f1')
    print_result(f1_df_by_root_sampling_method)
