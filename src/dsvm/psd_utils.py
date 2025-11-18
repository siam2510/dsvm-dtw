# src/dsvm/psd_utils.py

import numpy as np


def psd_shift_min_eig(K, eps_abs=1e-12):
    """
    최소 고윳값 기반 대각 시프트 (음수일 때만 보정)
        λ = -min_eig + eps_abs   (min_eig < 0 일 때만)
    """
    K = (K + K.T) / 2.0
    n = K.shape[0]

    w_min = float(np.linalg.eigvalsh(K)[0])
    lam = 0.0

    if w_min < 0.0:
        lam = -w_min + eps_abs
        K = K + lam * np.eye(n)

    K = (K + K.T) / 2.0
    return K, lam


def normalize_kernel_train(K, eps=1e-12):
    """
    학습 커널에 대해
        K_norm = D^{-1/2} K D^{-1/2}
    형태로 정규화하고, 대각 d = diag(K)를 반환.
    """
    K = (K + K.T) / 2.0
    d = np.clip(np.diag(K), eps, None) 
    Kn = K / np.sqrt(np.outer(d, d))
    np.fill_diagonal(Kn, 1.0)
    return Kn, d


def normalize_kernel_pair(K, dA, dB, eps=1e-12):
    """
    테스트-학습 커널 K에 대해
        K_norm = K / sqrt(dA) outer sqrt(dB)
    형태로 정규화.
    """
    DA = np.sqrt(np.maximum(dA, eps))
    DB = np.sqrt(np.maximum(dB, eps))
    return K / (DA[:, None] * DB[None, :])
