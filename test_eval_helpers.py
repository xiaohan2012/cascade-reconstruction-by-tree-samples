import pytest
import numpy as np


@pytest.fixture
def preds():
    return {0, 1, 2}

@pytest.fixture
def c():
    return np.array([-1, 0, 1, 2, -1])

@pytest.fixture
def obs():
    return [1]
