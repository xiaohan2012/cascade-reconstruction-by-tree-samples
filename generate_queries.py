# coding: utf-8

import time
import pickle as pkl
import os
import argparse

from tqdm import tqdm
from graph_tool import openmp_set_num_threads

from query_selection import (RandomQueryGenerator, EntropyQueryGenerator,
                             PRQueryGenerator, PredictionErrorQueryGenerator,
                             WeightedPredictionErrorQueryGenerator,
                             SamplingBasedGenerator,
                             MutualInformationQueryGenerator,
                             EarliestFirstOracle, LatestFirstOracle)
from simulator import Simulator
from joblib import Parallel, delayed
from graph_helpers import remove_filters, load_graph_by_name, get_edge_weights
from helpers import load_cascades
from sample_pool import TreeSamplePool
from random_steiner_tree.util import from_gt
from tree_stat import TreeBasedStatistics

parser = argparse.ArgumentParser(description='')
parser.add_argument('-g', '--graph', help='graph name')
parser.add_argument('-f', '--graph_suffix',
                    required=True,
                    help='suffix of graph name')
parser.add_argument('-w', '--weighted',
                    action='store_true',
                    help='if the graph is weighted')

parser.add_argument('-q', '--query_strategy',
                    choices={'random',
                             'pagerank',
                             'entropy',
                             'prediction_error',
                             'weighted-cond-ent',
                             'mutual-info',
                             'oracle-e',
                             'oracle-l'},
                    help='query strategy')
parser.add_argument('-c', '--cascade_dir',
                    help='directory of generated cascades')

# following applies to sampler approach
parser.add_argument('-n', '--n_queries', default=10, type=int,
                    help='number of queries')
parser.add_argument('-m', '--sampling_method', default='loop_erased', type=str,
                    choices={'loop_erased', 'cut', 'cut_naive'},
                    help='the steiner tree sampling method')
parser.add_argument('-r', '--root_sampler', type=str,
                    default='pagerank',
                    choices={'pagerank', 'random', 'true_root'},
                    help='the steiner tree sampling method')
parser.add_argument('-s', '--n_samples', default=100, type=int,
                    help='number of samples')
parser.add_argument('-i', '--incremental_cascade', action='store_true',
                    help='whether enable incremental cascade (probabilistic trimming) ' +
                    'for tree samples or not')

# specific to prediction error-based sampler
parser.add_argument('-p', '--min_proba', default=0.0, type=float,
                    help='(minimum) threshold used for pruning candidate nodes')
parser.add_argument('-e', '--num_estimation_nodes', default=None, type=int,
                    help='number of nodes used for error estimation')

parser.add_argument('-d', '--output_dir', default='outputs/queries', help='output directory')
parser.add_argument('-j', '--n_jobs', type=int, default=-1, help='number of workers in parallel')
parser.add_argument('-b', '--debug', action='store_true', help='if debug, use non-parallel')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='if verbose, verbose information is printed')

# specific to pagerank root sampler
parser.add_argument('--root_pagerank_noise', default=0.0, type=float,
                    help='the epsilon value for pagerank root sampling, the higher the more noisy')

args = parser.parse_args()

print("Args:")
print('-' * 10)
for k, v in args._get_kwargs():
    print("{}={}".format(k, v))
        
graph_name = args.graph
graph_suffix = args.graph_suffix

n_queries = args.n_queries
n_samples = args.n_samples
root_sampler = args.root_sampler

# output_dir = '{}/{}'.format(args.output_dir, graph_name)
output_dir = args.output_dir
sampling_method = args.sampling_method
incremental_cascade = args.incremental_cascade
query_strategy = args.query_strategy

# for prediction error-based query selector
min_proba = args.min_proba
num_estimation_nodes = args.num_estimation_nodes

g = load_graph_by_name(graph_name, weighted=args.weighted,
                       suffix=graph_suffix)

if query_strategy == 'random':
    strategy = (RandomQueryGenerator, {})
elif query_strategy == 'oracle-e':
    strategy = (EarliestFirstOracle, {})
elif query_strategy == 'oracle-l':
    strategy = (LatestFirstOracle, {})
elif query_strategy == 'pagerank':
    strategy = (PRQueryGenerator, {})
elif query_strategy == 'entropy':
    strategy = (EntropyQueryGenerator, {'method': 'entropy', 'root_sampler': root_sampler,
                                        'root_sampler_eps': args.root_pagerank_noise})
elif query_strategy == 'prediction_error':
    print("min_proba={}".format(min_proba))
    print("num_estimation_nodes={}".format(num_estimation_nodes))
    strategy = (PredictionErrorQueryGenerator, {'n_node_samples': None,
                                                'prune_nodes': True,
                                                'root_sampler': root_sampler,
                                                'root_sampler_eps': args.root_pagerank_noise,
                                                'min_proba': min_proba,
                                                'n_node_samples': num_estimation_nodes})
elif query_strategy == 'weighted-cond-ent':
    strategy = (WeightedPredictionErrorQueryGenerator,
                {'n_node_samples': None,
                 'prune_nodes': True,
                 'root_sampler': root_sampler,
                 'root_sampler_eps': args.root_pagerank_noise,
                 'min_proba': min_proba,
                 'n_node_samples': num_estimation_nodes})

elif query_strategy == 'mutual-info':
    strategy = (MutualInformationQueryGenerator,
                {'n_node_samples': None,
                 'prune_nodes': True,
                 'root_sampler': root_sampler,
                 'root_sampler_eps': args.root_pagerank_noise,
                 'min_proba': min_proba,
                 'n_node_samples': num_estimation_nodes})
else:
    raise ValueError('invalid strategy name')


def one_round(g, obs, c, c_path, q_gen_cls, param, q_gen_name, output_dir,
              sampling_method,
              incremental_cascade,
              n_samples,
              verbose):
    stime = time.time()
    c_id = os.path.basename(c_path).split('.')[0]
    d = output_dir
    outpath = os.path.join(d, c_id + '.pkl')

    # meta data such as running time, etc
    meta_outpath = os.path.join(d, c_id + '.meta.pkl')

    if os.path.exists(outpath):
        print("{} processed already, skip".format(c_path))
        return

    print('\nquerying {} started, using {}\n'.format(c_path, q_gen_name))

    gv = remove_filters(g)
    args = []  # sampling based method need a sampler to initialize

    weights = get_edge_weights(gv)
    gi = from_gt(gv, weights=weights)
    
    if issubclass(q_gen_cls, SamplingBasedGenerator):
        sampler = TreeSamplePool(
            gv,
            n_samples=n_samples,
            method=sampling_method,
            gi=gi,
            return_type='nodes',
            with_resampling=False,
            with_inc_sampling=incremental_cascade)
        args.append(sampler)
        param['error_estimator'] = TreeBasedStatistics(gv)
        
    q_gen = q_gen_cls(gv, *args, verbose=verbose, **param)
    sim = Simulator(gv, q_gen, gi=gi, print_log=verbose)

    qs = sim.run(n_queries, obs, c)

    time_cost = time.time() - stime

    if not os.path.exists(d):
        os.makedirs(d)
    print('\nquerying {} done: taking {:.4f} secs\n'.format(c_path, time_cost))
    pkl.dump(qs, open(outpath, 'wb'))

    meta_data = {'time_elapsed': time_cost}
    pkl.dump(meta_data, open(meta_outpath, 'wb'))

cascade_generator = load_cascades(args.cascade_dir)


# run!
if args.debug:
    print('====================')
    print('DEBUG MODE')
    print('====================')
    cls, param = strategy
    for path, tpl in tqdm(cascade_generator):
        one_round(g, tpl[0], tpl[1], path, cls, param, query_strategy, output_dir, sampling_method,
                  incremental_cascade,
                  n_samples,
                  args.verbose)
else:
    # prevent Parallel from hanging
    openmp_set_num_threads(1)
    
    Parallel(n_jobs=args.n_jobs)(delayed(one_round)(
        g, tpl[0], tpl[1], path, strategy[0], strategy[1],
        query_strategy, output_dir, sampling_method,
        incremental_cascade,
        n_samples,
        args.verbose)
                                 for path, tpl in tqdm(cascade_generator))
