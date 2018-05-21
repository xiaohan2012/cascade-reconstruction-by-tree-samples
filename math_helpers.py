import numpy as np


def hellinger_distance(a, b):
    return np.sqrt(1 - np.sum(np.sqrt(a * b)))
