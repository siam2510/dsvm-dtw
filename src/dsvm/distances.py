# src/dsvm/distances.py

import numpy as np
from scipy.spatial.distance import cdist
from dtaidistance import dtw


def compute_euclidean_distance_matrix(X):
    """
    pairwise Euclidean 거리 행렬.
    """
    X = np.asarray(X, dtype=float)
    dist_mat = cdist(X, X, metric="euclidean")
    return dist_mat


def compute_dtw_distance_matrix(X, window=None):
    """
    DTW 거리 행렬 (Sakoe-Chiba band 옵션).
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    dist_mat = np.zeros((n, n), dtype=float)

    used_fast = False
    used_slow = False

    for i in range(n):
        for j in range(i + 1, n):
            try:
                dist = dtw.distance_fast(X[i], X[j], window=window, use_pruning=True)
                if not used_fast:
                    print("Using fast DTW in distance matrix")
                    used_fast = True
            except AttributeError:
                dist = dtw.distance(X[i], X[j], window=window)
                if not used_slow:
                    print("Falling back to slow DTW in distance matrix")
                    used_slow = True
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    return dist_mat
