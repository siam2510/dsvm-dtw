# src/dsvm/dsvm_chart.py

import numpy as np
from sklearn.svm import SVC

from dsvm.kernels import custom_ecd_kernel, custom_dtw_kernel
from dsvm.psd_utils import (
    psd_shift_min_eig,
    normalize_kernel_train,
    normalize_kernel_pair,
)
from dsvm.stream_utils import construct_window_with_backfill


def single_rep_run(
    rep,
    X_stream_raw,
    X_pool,
    X_ref,
    y_ref,
    sigma2,
    N_w,
    m,
    mode,
    window,
):
    """
    기존 single_rep_run(args)와 기능 동일.
    rep마다 p_swk(t), t=0..m-1 리스트를 반환.
    """
    rng = np.random.default_rng(rep + 2000)
    buffer = []
    X_stream = rng.permutation(X_stream_raw)
    p_swk_list = []

    for t in range(m):
        z = X_stream[t]
        buffer.append(z)

        X_win = construct_window_with_backfill(
            np.array(buffer), t, X_pool, N_w, rng
        )
        y_win = np.ones(len(X_win))

        X_train = np.vstack([X_ref, X_win])
        y_train = np.concatenate([y_ref, y_win])

        # 학습 커널
        if mode == "ecd":
            K_train = custom_ecd_kernel(X_train, X_train, sigma2)
        elif mode == "dtw":
            K_train = custom_dtw_kernel(X_train, X_train, sigma2, window=window)
            K_train, lam_used = psd_shift_min_eig(K_train, eps_abs=1e-12)
        else:
            raise ValueError("mode는 'ecd' 또는 'dtw'만 가능합니다.")

        # 정규화
        K_train, d_train = normalize_kernel_train(K_train)

        clf = SVC(
            kernel="precomputed",
            C=1.0,
            shrinking=False,
            cache_size=1000,
        )
        clf.fit(K_train, y_train)

        # 테스트 커널
        if mode == "ecd":
            K_test = custom_ecd_kernel(X_win, X_train, sigma2)
        elif mode == "dtw":
            K_test = custom_dtw_kernel(X_win, X_train, sigma2, window=window)
        else:
            raise ValueError("mode는 'ecd' 또는 'dtw'만 가능합니다.")

        d_test = np.ones(len(X_win), dtype=float)
        K_test = normalize_kernel_pair(K_test, d_test, d_train)

        f_scores = clf.decision_function(K_test)
        g_scores = 1 / (1 + np.exp(-f_scores))
        p_swk = np.mean(g_scores)
        p_swk_list.append(p_swk)
        # 필요하면 여기 print 넣을 수 있음 (단, 멀티프로세싱이면 잘 안 보임)

    return p_swk_list
