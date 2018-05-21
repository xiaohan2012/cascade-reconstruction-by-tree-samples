import numpy as np
from graph_tool.spectral import laplacian

# @profile
def num_spanning_trees_dense(g, weights=None):
    """compute number of spanning trees

    it converts the sparse laplacian into a dense one
    so it's **memory inefficient**
    
    Param:
    ---------

    g: Graph
    """
    l = laplacian(g, weight=weights)
    l = l.todense()
    l0 = l[:-1, :-1]
    return np.linalg.det(l0)
