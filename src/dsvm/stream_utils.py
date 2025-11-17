# src/dsvm/stream_utils.py

import numpy as np


def construct_window_with_backfill(buffer, t, X_pool, N_w, rng):
    """
    buffer를 기반으로 길이 N_w의 윈도우 구성.
    앞부분이 부족하면 X_pool에서 채운 후 buffer를 이어붙인다.
    """
    if t < N_w - 1:
        n_pool = N_w - t - 1
        idx = rng.choice(len(X_pool), size=n_pool, replace=False)
        X_ic = X_pool[idx]
        X_win = np.vstack([X_ic, buffer])
    else:
        X_win = np.array(buffer[-N_w:])
    return X_win
