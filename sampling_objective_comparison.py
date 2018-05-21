import pandas as pd
import numpy as np
import argparse

from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
from tqdm import tqdm

from graph_tool import openmp_set_num_threads

from inference import infection_probability
from tree_stat import TreeBasedStatistics
from root_sampler import build_true_root_sampler
from helpers import infected_nodes, load_cascades
from graph_helpers import load_graph_by_name, get_edge_weights
from proba_helpers import cascade_probability_gt, ic_cascade_probability_gt
from random_steiner_tree.util import from_gt
from sample_pool import TreeSamplePool
from eval_helpers import precision_at_cascade_size


def run_with_or_without_resampling(g, cid, c, X, n_samples, sampling_method):
    gi = from_gt(g, get_edge_weights(g))
    infected = infected_nodes(c)
    y_true = np.zeros((len(c), ))
    y_true[infected] = 1
    X_set = set(X)
    mask = np.array([(i not in X_set) for i in range(len(c))])

    root_sampler = build_true_root_sampler(c)
    
    options = {
        'P': {'with_resampling': True, 'true_casacde_proba_func': cascade_probability_gt},
        'P_new': {'with_resampling': True, 'true_casacde_proba_func': ic_cascade_probability_gt},
        'no resampling': {'with_resampling': False}
    }

    ap_ans, p_ans = {}, {}
    for name, opt in options.items():
        sampler = TreeSamplePool(g, n_samples, sampling_method,
                                 gi=gi,
                                 return_type='nodes',
                                 **opt)
        sampler.fill(X, root_sampler=root_sampler)

        estimator = TreeBasedStatistics(g, sampler.samples)

        probas = infection_probability(g, X, sampler, estimator)

        ap_score = average_precision_score(y_true[mask], probas[mask])
        p_score = precision_at_cascade_size(y_true[mask], probas[mask])
        # print('with_resampling={}, AP score={}'.format(opt, score))
        ap_ans[name] = ap_score
        p_ans[name] = p_score
    ap_ans['cid'] = cid
    p_ans['cid'] = cid
    # print(ans)
    return ap_ans, p_ans

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--graph', help='graph name')
    parser.add_argument('-s', '--sampling_method', default='loop_erased', type=str,
                        choices={'loop_erased', 'cut'},
                        help='the steiner tree sampling method')
    parser.add_argument('-n', '--n_samples', default=1000, type=int,
                        help='number of samples')
    parser.add_argument('-j', '--n_jobs', default=-1, type=int,
                        help='number of parallel jobs')

    args = parser.parse_args()
    
    openmp_set_num_threads(1)

    graph_name = args.graph
    sampling_method = args.sampling_method
    n_samples = args.n_samples

    g_rev = load_graph_by_name(graph_name, weighted=True, suffix='_reversed')
    
    cs = load_cascades('cascade-weighted/{}-mic-s0.02-oleaves/'.format(graph_name))
        
    tuples_of_records = Parallel(n_jobs=args.n_jobs)(
        delayed(run_with_or_without_resampling)(g_rev, cid, c, X, n_samples, sampling_method)
        for cid, (X, c) in tqdm(cs, total=96))

    ap_records, p_records = zip(*tuples_of_records)
    ap_df = pd.DataFrame.from_records(ap_records)
    print('ap score:')
    print(ap_df.describe())

    pk_df = pd.DataFrame.from_records(p_records)
    print('precision@k score:')
    print(pk_df.describe())

    ap_df.to_pickle('outputs/sampling-objective-comparison/ap-g{}-s{}-n{}.pkl'.format(
        graph_name, sampling_method, n_samples))
    ap_df.describe().to_pickle('outputs/sampling-objective-comparison/ap-g{}-s{}-n{}.summary.pkl'.format(
        graph_name, sampling_method, n_samples))

    pk_df.to_pickle('outputs/sampling-objective-comparison/pk-g{}-s{}-n{}.pkl'.format(
        graph_name, sampling_method, n_samples))
    pk_df.describe().to_pickle('outputs/sampling-objective-comparison/pk-g{}-s{}-n{}.summary.pkl'.format(
        graph_name, sampling_method, n_samples))
