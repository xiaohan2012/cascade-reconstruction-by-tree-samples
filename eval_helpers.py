import numpy as np
import math
import os
import pickle as pkl
from copy import copy
from tqdm import tqdm

from graph_helpers import extract_nodes
from helpers import infected_nodes
from tree_stat import EPS
from sklearn.metrics import precision_score, average_precision_score, log_loss



class TooSmallCascadeError(Exception):
    pass


def infection_precision_recall(preds, c, obs):
    """
    given set of inferred infected nodes and ground truth, return precision and recall

    Args:
    set of ints: preds, set of predicted infections (besides observations)
    np.ndarray:  c, cascade
    bool: return_details, if the corret, false_positive, false_negative should be returned

    Return:
    float: precison
    float: recall
    """
    all_infs = set((c >= 0).nonzero()[0])
    remain_infs = all_infs - set(obs)
    preds -= set(obs)
    
    correct = preds.intersection(remain_infs)

    precision = len(correct) / len(preds)
    recall = len(correct) / len(remain_infs)

    return precision, recall


def precision_at_k(y, probas, k):
    pred_idx = np.argsort(probas)[::-1][:k]
    return precision_score(y[pred_idx], np.ones(k))


def precision_at_cascade_size(y, probas):
    k = (y > 0).sum()
    return precision_at_k(y, probas, k)


def top_k_infection_precision_recall(g, inf_probas, c, obs, k):
    """
    take the top k infections ordered by inf_probas, from high to low

    and then calculate the precision and recall w.r.t to obs
    """
    # rank and exclude the observed nodes first
    n2proba = {n: proba for n, proba in zip(extract_nodes(g), inf_probas)}
    inf_nodes = []
    for i in sorted(n2proba, key=n2proba.__getitem__, reverse=True):
        if len(inf_nodes) == k:
            break
        if i not in obs:
            inf_nodes.append(i)
            
    return infection_precision_recall(set(inf_nodes), c, obs)


def aggregate_scores_over_cascades_by_methods(cascades,
                                              method_labels,
                                              query_dir_ids,
                                              inf_dir_ids,
                                              n_queries,
                                              inf_result_dirname,
                                              query_dirname,
                                              eval_method,
                                              eval_with_mask,
                                              iter_callback=None,
                                              every=1):
    """
    each element in `method_labels` uniquely identifies one experiment

    Returns: method_name -> [n_experiments x n_queries]
    """
    assert len(method_labels) == len(query_dir_ids) == len(inf_dir_ids)
    # dict of key -> [n_experiments x n_queries]
    scores_by_method = {}
    for l in method_labels:
        scores_by_method[l] = []
    
    c_paths = []  # track the order
    for c_path, tpl in tqdm(cascades):
        obs, c = tpl[:2]
        
        obs = set(obs)
        c_paths.append(c_path)
        infected = (c >= 0).nonzero()[0]
        # infected_set = set(infected)
        # print('infection size', len(infected_set))
        # labels for nodes, 1 for infected, 0 for uninfected
        y_true = np.zeros((len(c), ))
        y_true[infected] = 1
        
        for method_label, query_dir, inf_dir in zip(method_labels, query_dir_ids, inf_dir_ids):
            cid = os.path.basename(c_path).split('.')[0]

            # load infection probabilities
            inf_probas_path = os.path.join(
                inf_result_dirname,
                inf_dir,
                '{}.pkl'.format(cid))

            try:
                inf_probas_list = pkl.load(open(inf_probas_path, 'rb'))
            except EOFError:
                print('**EOFError, inf_probas_path=', inf_probas_path)
                print("**WARNING**: ignore corrupted file")
                continue
                # raise
            # print('inf_probas_path', inf_probas_path)
            # print('inf_probas_list', inf_probas_list)

            # load queries
            query_path = os.path.join(
                query_dirname, query_dir, '{}.pkl'.format(cid))

            # print('query_path', query_path)
            queries = pkl.load(open(query_path, 'rb'))[0]

            scores = get_scores_by_queries(queries,
                                           inf_probas_list,
                                           c, obs,
                                           eval_method,
                                           every=every,
                                           eval_with_mask=eval_with_mask,
                                           iter_callback=iter_callback)
            # if method_label == 'prederror':
            # print(scores)
            scores_by_method[method_label].append(scores)
        # print('---'*10)
    return scores_by_method


def eval_probas(c, X, probas):
    infected = infected_nodes(c)
    y_true = np.zeros((len(c), ))
    y_true[infected] = 1
    X_set = set(X)
    mask = np.array([(i not in X_set) for i in range(len(c))])
    
    ap_score = average_precision_score(y_true[mask], probas[mask])
    p_score = precision_at_cascade_size(y_true[mask], probas[mask])
    return {'ap': ap_score, 'pk': p_score}


def mean_average_precision(y_true, p_pred):
    idx = np.argsort(p_pred)[::-1]
    y = y_true[idx]
    map_score = np.mean([y[:i+1].sum() / (i+1) for i in y.nonzero()[0]])
    return map_score


def mean_reciprical_rank(y_true, p_pred):
    idx = np.argsort(p_pred)[::-1]
    # p = p_pred[idx]
    y = y_true[idx]
    k = y_true.sum()

    scores = 1 / (1 + (y == 1).nonzero()[0])
    M = 1 / np.arange(1, k+1)
    return scores.sum() / M.sum()


def get_scores_by_queries(qs, probas, c, obs,
                          eval_method,
                          every=1,
                          eval_with_mask=True,
                          iter_callback=None,
                          node_score_callback=None,
                          **kwargs):
    inf_nodes = set(infected_nodes(c))
    cascade_size = len(inf_nodes)
    y_true = np.zeros((len(c), ))
    y_true[infected_nodes(c)] = 1
    obs_inc = copy(set(obs))

    inf_obs = set(obs)
    uninf_obs = set()

    scores_list = []

    for i_iter, query in enumerate(qs):
        if c[query] == -1:
            uninf_obs.add(query)
        else:
            inf_obs.add(query)
            
        obs_inc.add(query)

        scores = None  # ugly code..
        if i_iter % every == 0:
            # print(i_iter)
            i = int(i_iter / every)
            if i >= len(probas):
                break
            inf_probas = probas[i + 1]  # offset +1 because of the initial probas

            if callable(iter_callback):
                # print('iter_callback: ON')
                iter_callback(inf_probas, inf_obs, uninf_obs)
                
            if eval_with_mask:
                mask = np.array([(i not in obs_inc) for i in range(len(c))])
            else:
                mask = np.ones(len(c), dtype=np.bool)

            try:
                if eval_method == 'ap':
                    score = average_precision_score(y_true[mask], inf_probas[mask])
                elif eval_method == 'p_at_k':
                    k = kwargs['k']
                    score = precision_at_k(y_true[mask], inf_probas[mask], k)
                elif eval_method == 'n':
                    score = precision_at_k(y_true[mask], inf_probas[mask], cascade_size)
                    score *= cascade_size
                elif eval_method == 'p@k':
                    k = len(inf_nodes - set(obs_inc))
                    score = precision_at_k(y_true[mask], inf_probas[mask], k)
                elif eval_method == 'entropy':
                    p = inf_probas[mask]
                    p = p[(p != 0) & (p != 1)]
                    score = (-(p * np.log(p) + (1-p) * np.log(1-p))).sum()
                elif eval_method == 'map':
                    score = mean_average_precision(y_true[mask], inf_probas[mask])
                elif eval_method == 'mrr':
                    score = mean_reciprical_rank(y_true[mask], inf_probas[mask])
                elif eval_method == 'ratio_discovered_inf':
                    # print('len(obs_inc)', len(obs_inc))
                    # print('num. discovered', sum(1 for o in obs_inc if c[o] >= 0))
                    score = sum(1 for o in obs_inc if c[o] >= 0) / len(inf_nodes)
                elif eval_method == 'l1':
                    # score = np.abs(y_true[mask] - inf_probas[mask]).mean()
                    scores = np.abs(y_true[mask] - inf_probas[mask])
                elif eval_method == 'l2':
                    scores = np.power(y_true[mask] - inf_probas[mask], 2)
                elif eval_method == 'cross_entropy':
                    y = y_true[mask]
                    y = np.clip(y, EPS, 1 - EPS)  # clip it between (epsilon, 1 - epsilon)

                    p = inf_probas[mask]
                    p = np.clip(p, EPS, 1 - EPS)

                    y_inv = 1 - y
                    p_inv = 1 - p

                    # score = (- y * np.log(p)).mean()
                    # score -= (y_inv * np.log(p_inv)).mean()
                    score0 = -(y * np.log(p))
                    score1 = -(y_inv * np.log(p_inv))

                    if node_score_callback is not None:
                        node_score_callback([score0, score1])
                    scores = score0 + score1
                    # print(score)
                else:
                    raise ValueError('not valid eval method {}'.format(eval_method))

                if scores is not None:
                    if node_score_callback is not None:
                        node_score_callback(scores)
                    score = scores.sum()

            except FloatingPointError:
                score = 0
            
            if np.isnan(score):
                score = 0
            scores_list.append(score)
    assert len(scores_list) == math.ceil(len(qs) / every), \
        '{} != {}'.format(len(scores_list), math.ceil(len(qs) / every),)
    return scores_list
