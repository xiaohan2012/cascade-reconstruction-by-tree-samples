# cython: linetrace=True

import numpy as np
from libc.math cimport log

cdef double EPS = 1e-8

cdef bint close_to_zero_or_one(double v):
    return (abs(v) < EPS) or (abs(1.0 - v) < EPS)

# cdef bint close_to_one(double v):
#     return abs(1.0 - v) < EPS

cpdef matching_trees_cython(T, int node, int value):
    """
    T: list of set of ints, list of trees represented by nodes
    node: node to filter
    value: value to filter
    """
    if value == 1:  # infected
        return [t for t in T if node in t]
    else:  # uninfected
        return [t for t in T if (node not in t)]

def matching_trees(T, node_values):
    """
    T: list of set of ints, list of trees represented by nodes
    node: node to filter
    value: value to filter
    """
    def predicate(t):
        for n, v in node_values.items():
            if v == 1:
                if n not in t:
                    return False
            else:
                if n in t:
                    return False
        return True
    return [t for t in T if predicate(t)]

    # if value == 1:  # infected
    #     return [t for t in T if node in t]
    # else:  # uninfected
    #     return [t for t in T if node not in t]

# @profile
cpdef prediction_error(int q, int y_hat, T, hidden_nodes):
    # filter T by (q, y_hat)
    sub_T = matching_trees_cython(T, q, y_hat)
    cdef double p, error = 0.0, N = len(sub_T)

    if N > 0:  # avoid ZeroDivisionError
        for u in hidden_nodes:
            p = len(matching_trees_cython(sub_T, u, 0)) / N
            if not close_to_zero_or_one(p):
                error -= (p * log(p) + (1-p) * log(1-p))
    return error

# @profile
def query_score(int q, T, hidden_nodes):
    assert q not in hidden_nodes
    cdef double p, score = 0
    cdef int y_hat
    for y_hat in [0, 1]:
        p = <double>len(matching_trees_cython(T, q, y_hat)) / len(T)
        error = prediction_error(q, y_hat, T, hidden_nodes)
        score += p * prediction_error(q, y_hat, T, hidden_nodes)

    return score
