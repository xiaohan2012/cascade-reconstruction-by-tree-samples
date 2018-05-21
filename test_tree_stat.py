import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_eq_np, assert_almost_equal
from graph_tool import Graph
from scipy.stats import entropy

from tree_stat import TreeBasedStatistics


@pytest.fixture
def trees():
    """
    0 0 1 0 0 1
    1 0 0 1 0 0
    1 0 0 1 1 0
    0 0 0 1 1 1
    1 0 1 1 0 0

sum 3 0 2 4 2 2
    """
    return [
        {2, 5},
        {0, 3},
        {0, 3, 4},
        {3, 4, 5},
        {0, 2, 3}
    ]


@pytest.fixture
def new_trees():
    return [
        {0, 1},
        {0, 4}
    ]


@pytest.fixture
def g():
    g = Graph(directed=False)
    g.add_vertex(6)
    return g


@pytest.fixture
def stat(g, trees):
    return TreeBasedStatistics(g, trees)


@pytest.mark.parametrize("targets", [None, set(range(6)), list(range(6))])
def test_unconditional_count_and_proba(stat, trees, targets):
    arr_c = np.array([3, 0, 2, 4, 2, 2])
    arr_p = arr_c / len(trees)
    assert_eq_np(stat.unconditional_count(targets),
                 arr_c)

    assert_eq_np(stat.unconditional_proba(targets),
                 arr_p)


@pytest.mark.parametrize("targets", [None, set(range(6)), list(range(6))])
def test_filter_out_extreme_targets(stat, trees, targets):
    # [3, 0, 2, 4, 2, 2] --> [2, 0, 2, 1, 2, 2] --> [2/5, 0, 2/5, 1/5, 2/5, 2/5]
    filtered_targets = stat.filter_out_extreme_targets(targets,
                                                       min_value=2/len(trees))  # 2/5
    assert set(filtered_targets) == set()

    filtered_targets = stat.filter_out_extreme_targets(targets,
                                                       min_value=1/len(trees))  # 1/5
    assert set(filtered_targets) == {0, 2, 4, 5}

    filtered_targets = stat.filter_out_extreme_targets(targets,
                                                       min_value=0/len(trees))  # 0
    assert set(filtered_targets) == {0, 2, 3, 4, 5}


def test_count_and_proba(stat, trees):
    targets = list(range(1, 6))
    query = 0

    arr_c0 = np.array([0, 1, 1, 1, 2])
    arr_c1 = np.array([0, 1, 3, 1, 0])

    assert_eq_np(stat.count(query, condition=0, targets=targets),
                 arr_c0)
    assert_eq_np(stat.count(query, condition=1, targets=targets),
                 arr_c1)

    assert_eq_np(stat.proba(query, condition=0, targets=targets),
                 arr_c0 / 2)
    assert_eq_np(stat.proba(query, condition=1, targets=targets),
                 arr_c1 / 3)


def test_update_trees(stat, new_trees):
    """
    1 1 0 0 0 0 (new*)
    1 0 0 1 0 0
    1 0 0 1 1 0
    1 0 0 0 1 0 (new*)
    1 0 1 1 0 0

sum 5 1 1 3 2 0
    """
    stat.update_trees(new_trees, {0: 1})

    query = 1
    targets = [2, 3, 4, 5]
    arr_c0 = [1, 3, 2, 0]
    arr_c1 = [0, 0, 0, 0]

    assert_eq_np(stat.count(query, condition=0, targets=targets),
                 arr_c0)
    assert_eq_np(stat.count(query, condition=1, targets=targets),
                 arr_c1)


def test_update_trees_multiple_nodes_update(stat, new_trees):
    """
    1 1 0 0 0 0 (new*)
    1 0 0 1 0 0
    1 0 0 0 0 0 (new*)
    1 0 0 0 1 0 (new*)
    1 0 1 1 0 0

sum 5 1 1 3 2 0
    """

    new_trees.append({1})
    stat.update_trees(new_trees, {0: 1, 4: 0})

    query = 1
    targets = [2, 3, 5]
    arr_c0 = [1, 2, 0]
    arr_c1 = [0, 0, 0]

    assert_eq_np(stat.count(query, condition=0, targets=targets),
                 arr_c0)
    assert_eq_np(stat.count(query, condition=1, targets=targets),
                 arr_c1)


def test_update_trees_insufficient_trees(stat, new_trees):
    with pytest.raises(AssertionError):
        # insufficient length
        stat.update_trees(new_trees[:1], {0: 1})


@pytest.fixture
def trees1():
    return [
        {0, 1, 2},
        {1, 2},
        {1, 2, 3},
        {1, 2, 3, 4},
    ]


@pytest.fixture
def stat1(g, trees1):
    return TreeBasedStatistics(g, trees1)


def test_prediction_error(stat1):
    actual = stat1.prediction_error(0, 0, [3, 4])
    expected = entropy([1/3, 2/3]) * 2
    assert_almost_equal(actual, expected)


def test_query_score(stat1):
    actual = stat1.query_score(0, [3, 4])
    expected = entropy([1/3, 2/3]) * 2 * 3/4  # + error = 0 for state=1
    assert_almost_equal(actual, expected)
