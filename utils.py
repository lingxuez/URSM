## Helper functions
##
## Copyright Lingxue Zhu (lzhu@cmu.edu).
## All Rights Reserved.

import numpy as np

def simplex_proj(y, min_y):
    """
    project a vector y onto the simplex, such that
     y >= min_y and sum(y) = 1
    Reference: Wang (2013), "Projection onto the probability simplex:
        An efficient algorithm with a simple proof, and an application"
    -- y: an N x 1 vector to be projected
    """
    p = len(y)
    sorted_y = sorted(y, reverse=True)

    cumsum = 0 ## cumulative sum
    (iBest, rouBest) = (-1, -1) ## the largest i such that rou>min_y
    for i in xrange(p):
        cumsum += sorted_y[i]
        rou = sorted_y[i] + (1 - (p-i-1)*min_y - cumsum) / float(i+1)
        if rou > min_y:
            iBest = i
            rouBest = rou

    lamb = rouBest - sorted_y[iBest]
    proj_y = np.maximum(np.array(y)+lamb, min_y)

    return proj_y


def std_row(B):
    """
    Standardize matrix to have row sum to 1
    """
    return np.true_divide(B, B.sum(axis=1)[:, np.newaxis])



