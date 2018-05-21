import pytest
import numpy as np
from graph_tool import Graph
from eval_helpers import infection_precision_recall, \
    top_k_infection_precision_recall

@pytest.fixture
def preds():
    return {0, 1, 2}

@pytest.fixture
def c():
    return np.array([-1, 0, 1, 2, -1])

@pytest.fixture
def obs():
    return [1]


def test_infection_precision_recall(preds, c, obs):
    prec, rec = infection_precision_recall(
        preds, c, obs)
    assert prec == 0.5
    assert rec == 0.5


def test_top_k_infection_precision_recall(c, obs):
    g = Graph(directed=False)
    g.add_vertex(4)
    
    probas = [0.8, 0.9, 1.0, 0.5, 0.4]
    prec, rec = top_k_infection_precision_recall(g, probas, c, obs, k=1)
    assert prec == 1.0
    assert rec == 0.5
    
    prec, rec = top_k_infection_precision_recall(g, probas, c, obs, k=2)
    assert prec == 0.5
    assert rec == 0.5

    prec, rec = top_k_infection_precision_recall(g, probas, c, obs, k=3)
    assert prec == 2/3
    assert rec == 1.0
