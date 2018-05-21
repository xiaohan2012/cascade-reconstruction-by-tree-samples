import pytest
from core1 import matching_trees_cython, matching_trees, prediction_error, query_score
from scipy.stats import entropy

@pytest.fixture
def T():
    return [
        {0, 1, 2},
        {1, 2},
        {1, 2, 3},
        {1, 2, 3, 4},
    ]


@pytest.mark.parametrize("func", [matching_trees_cython])
def test_matching_trees(T, func):
    assert func(T, 0, 1) == [{0, 1, 2}]
    assert func(T, 1, 0) == []
    assert func(T, 4, 0) == T[:3]


@pytest.mark.parametrize("func", [matching_trees])
def test_matching_trees_new(T, func):
    assert func(T, {0: 1}) == [{0, 1, 2}]
    assert func(T, {1: 1, 3: 1}) == [{1, 2, 3}, {1, 2, 3, 4}]
    assert func(T, {1: 1, 3: 0}) == [{0, 1, 2}, {1, 2}]
    assert func(T, {4: 0}) == T[:3]


def test_prediction_error(T):
    error = prediction_error(0, 0, T, [3, 4])
    expected = entropy([1/3, 2/3]) * 2
    assert error == expected

    error = prediction_error(0, 1, T, [3, 4])
    assert error == 0


def test_query_score(T):
    score = query_score(0, T, [3, 4])
    expected = entropy([1/3, 2/3]) * 2 * 3/4
    assert score == expected
