# coding: utf-8

import os
import pickle as pkl
import numpy as np
import matplotlib
import argparse
matplotlib.use('pdf')

from matplotlib import pyplot as plt

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from helpers import load_cascades
from graph_helpers import load_graph_by_name
from eval_helpers import aggregate_scores_over_cascades_by_methods, precision_at_cascade_size
from test_helpers import check_probas_so_far
from viz_helpers import set_cycler


plt.style.use('paper')
np.seterr(divide='raise', invalid='raise')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--graph_name', help='graph name')
    parser.add_argument('-d', '--data_id', help='data id (e.g, "grqc-mic-o0.1")')

    parser.add_argument('--use_cache', action='store_true',
                        help='use evaluation result from cache or not')
    
    # eval method
    parser.add_argument('-e', '--eval_method',
                        choices=('ap', 'auc', 'p@k', 'entropy', 'map', 'mrr', 'n',
                                 'ratio_discovered_inf', 'l1', 'l2', 'cross_entropy'),
                        help='evalulation method')
    parser.add_argument('--eval_with_mask',
                        action="store_true",
                        help='whether evaluate with masks or not. If True, queries and obs are excluded')
    
    # root directory names
    parser.add_argument('-c', '--cascade_dirname', help='cascade directory name')
    parser.add_argument('--inf_dirname', help='')
    parser.add_argument('--query_dirname', default='queries', help='default dirname for query result')

    # query and inf dir ids
    parser.add_argument('-q', '--query_dir_ids', required=True,
                        help='list of query directory ids separated by ","')
    parser.add_argument('-i', '--inf_dir_ids', required=True,
                        help="""
list of infection probas directory ids separated by ","
why this? refer to plot_inference_using_weighted_vs_unweighted.sh""")
    
    parser.add_argument('-n', '--n_queries', type=int, help='number of queries to show')
    parser.add_argument('--every', type=int, help='evaluate every `every` iterations')
    parser.add_argument('--plot_step', type=int, help='plot every `plot_step` step')

    parser.add_argument('--check', action='store_true', help='if checking samples at each eval iteration')
    
    parser.add_argument('-s', '--sampling_method', help='')
    parser.add_argument('-l', '--legend_labels',
                        help='list of labels to show in legend separated  by ","')
    parser.add_argument('-f', '--fig_name', help='figure name')

    args = parser.parse_args()

    print("Args:")
    print('-' * 10)
    for k, v in args._get_kwargs():
        print("{}={}".format(k, v))
    
    inf_result_dirname = 'outputs/{}/{}/{}'.format(args.inf_dirname,
                                                   args.data_id,
                                                   args.sampling_method)
    query_dirname = 'outputs/{}/{}/{}'.format(args.query_dirname,
                                              args.data_id,
                                              args.sampling_method)

    print('summarizing ', inf_result_dirname)
    # if n_queries is too large, e.g, 100,
    # we might have no hidden infected nodes left and average precision score is undefined
    n_queries = args.n_queries

    g = load_graph_by_name(args.graph_name)

    query_dir_ids = list(map(lambda s: s.strip(), args.query_dir_ids.split(',')))
    if args.legend_labels is not None:
        labels = list(map(lambda s: s.strip(), args.legend_labels.split(',')))
    else:
        labels = query_dir_ids
    print('query_dir_ids:', query_dir_ids)

    if args.eval_with_mask:
        pkl_dir = 'eval_result/{}'.format(args.eval_method)
    else:
        pkl_dir = 'eval_result/{}-no-mask'.format(args.eval_method)

    print('pkl dir', pkl_dir)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)

    if not args.use_cache:
        inf_dir_ids = list(map(lambda s: s.strip(), args.inf_dir_ids.split(',')))
        print('inf_dir_ids:', inf_dir_ids)
        print('labels:', labels)
        
        cascades = load_cascades('{}/{}'.format(args.cascade_dirname, args.data_id))

        assert n_queries > 0, 'non-positive num of queries'

        # if args.eval_method == 'ap':
        #     eval_func = average_precision_score
        # elif args.eval_method == 'auc':
        #     eval_func = roc_auc_score
        # elif args.eval_method == 'precision_at_hidden_inf':
        #     print('precision_at_cascade_size')
        #     eval_func = precision_at_cascade_size
        # else:
        #     raise NotImplementedError(args.eval_method)
        
        scores_by_method = aggregate_scores_over_cascades_by_methods(
            cascades,
            labels,
            query_dir_ids,
            inf_dir_ids,
            n_queries,
            inf_result_dirname,
            query_dirname,
            args.eval_method,
            args.eval_with_mask,
            every=args.every,
            iter_callback=(check_probas_so_far if args.check else None))

        # make shape match
        max_len = max(len(r) for method in labels for r in scores_by_method[method])
        print('num. evaluation points: ', max_len)
        for method in labels:
            assert len(scores_by_method[method]) > 0, 'no scores available for {}'.format(method)
            for r in scores_by_method[method]:
                # for each row, pad to max_len
                val = r[-1]  # value to pad
                for i in range(max_len - len(r)):
                    # r.append(np.nan)
                    r.append(val)
                assert len(r) == max_len, "len(r)={}, r={}".format(len(r), r)

        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir)
        print('dumping eval result to ', pkl_dir + '/'+args.fig_name + '.pkl')
        pkl.dump(scores_by_method,
                 open('{}/{}.pkl'.format(pkl_dir, args.fig_name), 'wb'))
    else:
        print('load from cache')
        scores_by_method = pkl.load(open('{}/{}.pkl'.format(pkl_dir, args.fig_name), 'rb'))
        max_len = max(len(r) for method in labels for r in scores_by_method[method])
            
    # plotting
    plt.clf()

    fig, ax = plt.subplots(figsize=(5, 4))
    set_cycler(ax)

    # print('scores_by_method:', scores_by_method)
    min_y, max_y = 1, 0

    # print('max_len', max_len)
    # print('every', args.every)
    # print('product', max_len * args.every)
    # print('n_queries', n_queries)
    n_queries = min(n_queries, max_len * args.every)
    # print('n_queries (after)', n_queries)
    x = np.arange(0, n_queries, args.every)
    # print('x (original)', x)
    for method in labels:
        print('method', method)
        scores = np.array(scores_by_method[method], dtype=np.float32)

        # scores[np.isnan(scores)] = 0
        # mean_scores = np.mean(scores, axis=0)
        perc25 = np.percentile(scores, 25, axis=0)
        perc50 = np.percentile(scores, 50, axis=0)
        perc75 = np.percentile(scores, 75, axis=0)

        # print(x.shape)
        # print(mean_scores.shape)
        # print('x (new)', x[::args.plot_step])
        # print('y (new)', perc50[::args.plot_step])
        l = ax.plot(x[::args.plot_step], perc50[::args.plot_step])
        print('score', perc50[::args.plot_step])
        # print(l)
        ax.fill_between(x[::args.plot_step],
                        perc25[::args.plot_step],
                        perc75[::args.plot_step],
                        facecolor=l[0].get_color(),
                        lw=0,
                        alpha=0.5)

        min_y = min([min_y, np.min(perc50)])
        max_y = max([max_y, np.max(perc50)])
        # ax.hold(True)
    # ax.legend(labels, loc='best', ncol=1)
    # ax.xaxis.label.set_fontsize(20)
    # ax.yaxis.label.set_fontsize(20)
    # ax.set_ylim(min_y - 0.01, max_y + 0.01)
    # ax.set_ylim(0.01, 0.1)
    ax.set_xlabel('num. of queries')
    ax.set_ylabel(args.eval_method)
    
    fig.tight_layout()

    # plt.ylim(0.2, 0.8)
    if args.eval_with_mask:
        dir_suffix = ''
    else:
        dir_suffix = '-no-mask'

    dirname = 'figs/{}'.format(args.eval_method + dir_suffix)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    output_path = '{}/{}.pdf'.format(dirname, args.fig_name)
    fig.savefig(output_path)
    print('saved to {}'.format(output_path))
