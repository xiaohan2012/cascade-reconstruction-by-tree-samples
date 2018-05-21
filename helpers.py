import numpy as np
import pickle as pkl
import errno
import os
import signal

from glob import glob
from scipy.spatial.distance import cdist
from functools import wraps


def load_cascades(dirname):
    for p in glob(dirname+'/*.pkl'):
        yield p, pkl.load(open(p, 'rb'))


def cascade_source(c):
    return np.nonzero((c == 0))[0][0]


def infected_nodes(c):
    return np.nonzero((c >= 0))[0]


def l1_dist(probas1, probas2):
    return cdist([probas1],
                 [probas2],
                 'minkowski', p=1.0)[0, 0]


def cascade_info(obs, c):
    print('source: {}'.format(cascade_source(c)))
    print('|casdade|: {}'.format(len(infected_nodes(c))))
    print('|observed nodes|: {}'.format(len(obs)))


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)
    return decorator


def sampling_weights_by_order(length):
    w = 1 / (np.arange(10) + 1)[::-1]
    w /= w.sum()
    return w
