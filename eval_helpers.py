import numpy as np
import math
import os
import pickle as pkl
from tqdm import tqdm

from sklearn.metrics import average_precision_score
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
