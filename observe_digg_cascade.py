# coding: utf-8

import os
import pickle as pkl
from glob import glob

from itertools import product
from tqdm import tqdm
from cascade_generator import observe_cascade
from helpers import cascade_source, makedir_if_not_there


obs_fraction = 0.5
obs_method = 'uniform'

dirname = 'data/digg/'
num_repeats_per_cascade = 10

obs_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

for obs_fraction in tqdm(obs_fractions):
    output_dir = 'cascade/digg-o{}-om{}'.format(obs_fraction, obs_method)

    makedir_if_not_there(output_dir)

    for i, (path, _) in enumerate(
        product(glob(os.path.join(dirname, '*.pkl')),
                range(num_repeats_per_cascade))):
        c = pkl.load(open(path, 'rb'))
        source = cascade_source(c)
        obs = observe_cascade(c, source, obs_fraction, method=obs_method, source_includable=True)
        output_path = os.path.join(output_dir, '{}.pkl'.format(i))

        pkl.dump((obs, c), open(output_path, 'wb'))
