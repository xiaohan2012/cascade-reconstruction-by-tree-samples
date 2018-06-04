import numpy as np
import os
import pickle as pkl
from tqdm import tqdm

from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
from glob import glob


def evaluate_score(input_dir, output_dir, score_func):
    scores = []
    for input_path in tqdm(glob(input_dir + '*.pkl')):
        basename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, basename)
        obs, c = pkl.load(open(input_path, 'rb'))
        n = len(c)
        inf_probas = pkl.load(open(output_path, 'rb'))['inf_probas']
        
        hidden = list(set(np.arange(n)) - set(obs))
        y_true = np.array((c >= 0), dtype=np.double)
        
        scores.append(score_func(y_true[hidden], inf_probas[hidden]))
    return scores


def eval_map(*args):
    return evaluate_score(*args, score_func=average_precision_score)


def evaluate_edge_prediction(g, true_edges, pred_edge_freq, eval_func):
    edge_true_vect = g.new_edge_property('float')
    edge_true_vect.set_value(0)
    for u, v in true_edges:
        edge_true_vect[g.edge(u, v)] = 1
        
    edge_pred_vect = g.new_edge_property('float')
    edge_pred_vect.set_value(0)
    for (u, v), f in pred_edge_freq.items():
        edge_pred_vect[g.edge(u, v)] = f
    return eval_func(edge_true_vect.a, edge_pred_vect.a)


def eval_edge_parallel_task(g, input_path, output_dir, score_func):
    """one task to send to joblib.Parallel
    """
    basename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, basename)
    
    _, _, true_edges = pkl.load(open(input_path, 'rb'))
    edge_freq = pkl.load(open(output_path, 'rb'))['edge_freq']
    return evaluate_edge_prediction(g, true_edges, edge_freq, score_func)
    

def evaluate_edge_in_batch(g, input_dir, output_dir, score_func):
    scores = Parallel(n_jobs=-1)(
        delayed(eval_edge_parallel_task)(g, input_path, output_dir, score_func)
        for input_path in tqdm(glob(input_dir + '*.pkl'))
    )
    
    # for input_path in tqdm(glob(input_dir + '*.pkl')):
    #     basename = os.path.basename(input_path)
    #     output_path = os.path.join(output_dir, basename)

    #     _, _, true_edges = pkl.load(open(input_path, 'rb'))
    #     edge_freq = pkl.load(open(output_path, 'rb'))['edge_freq']
    #     scores.append(evaluate_edge_prediction(g, true_edges, edge_freq, score_func))

    return scores


def eval_edge_map(*args):
    return evaluate_edge_in_batch(*args, score_func=average_precision_score)
