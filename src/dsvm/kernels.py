# src/dsvm/kernels.py

import numpy as np
from scipy.spatial.distance import cdist
from dtaidistance import dtw


def custom_dtw_kernel(A, B, sigma2, window=None):
    """
    DTW-RBF kernel:
        K(i,j) = exp(-DTW(A[i], B[j])^2 / sigma^2)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n, m = len(A), len(B)
    dist2 = np.zeros((n, m), dtype=float)

    used_fast = False
    used_slow = False

    for i in range(n):
        for j in range(m):
            try:
                d = dtw.distance_fast(A[i], B[j], window=window, use_pruning=True)
                if not used_fast:
                    # print("Using fast DTW (C-backend)")
                    used_fast = True
            except AttributeError:
                d = dtw.distance(A[i], B[j], window=window)
                if not used_slow:
                    # print("Falling back to slow DTW (pure Python)")
                    used_slow = True
            dist2[i, j] = d * d

    return np.exp(-dist2 / sigma2)


def custom_ecd_kernel(A, B, sigma2):
    """
    Euclidean-distance RBF kernel:
        K(i, j) = exp(-||A[i] - B[j]||^2 / sigma^2)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    D2 = cdist(A, B, metric="sqeuclidean")
    return np.exp(-D2 / sigma2)
