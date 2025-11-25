from pathlib import Path
import numpy as np
import pandas as pd
from dtaidistance import dtw
import matplotlib.pyplot as plt

def compute_class_mean_curves(df, label_col=0):
    """
    Compute mean curves for each class label in a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw ECG dataframe with shape (n_samples, 1 + T)
        First column is class label.
    label_col : int
        Column index that contains class labels.

    Returns
    -------
    dict
        {label: mean_curve (pd.Series)}
    """
    labels = sorted(df[label_col].unique())
    mean_curves = {}

    for label in labels:
        samples = df[df[label_col] == label].iloc[:, 1:]  # exclude label column
        mean_curves[label] = samples.mean(axis=0)

    return mean_curves

def load_ecg5000(raw_dir: Path):
    """
    Load ECG5000 TRAIN/TEST files from raw_dir.

    Parameters
    ----------
    raw_dir : Path
        Directory that contains ECG5000_TRAIN.txt and ECG5000_TEST.txt.

    Returns
    -------
    train : pandas.DataFrame
        Raw training set (label + series).
    test : pandas.DataFrame
        Raw test set (label + series).
    """
    train_path = raw_dir / "ECG5000_TRAIN.txt"
    test_path = raw_dir / "ECG5000_TEST.txt"

    train = pd.read_csv(train_path, sep=r"\s+", engine="python", header=None)
    test = pd.read_csv(test_path,  sep=r"\s+", engine="python", header=None)

    return train, test

def load_wafer(raw_dir: Path):
    """
    Load Wafer TRAIN/TEST files from raw_dir.

    Parameters
    ----------
    raw_dir : Path
        Directory that contains Wafer_TRAIN.txt and Wafer_TEST.txt.

    Returns
    -------
    train : pandas.DataFrame
        Raw training set (label + series).
    test : pandas.DataFrame
        Raw test set (label + series).
    """
    train_path = raw_dir / "Wafer_TRAIN.txt"
    test_path = raw_dir / "Wafer_TEST.txt"

    train = pd.read_csv(train_path, sep=r"\s+", engine="python", header=None)
    test = pd.read_csv(test_path,  sep=r"\s+", engine="python", header=None)

    return train, test


def split_in_out(train: pd.DataFrame,
                 test: pd.DataFrame,
                 incontrol_label: int = 1,
                 seed: int = 1):
    """
    Split ECG5000 into in-control and out-of-control sets.

    - in-control: label == incontrol_label
    - out-of-control: label != incontrol_label
    - train_incontrol and test_incontrol are re-shuffled halves of
      combined in-control samples, following your current logic.

    Returns
    -------
    train_incontrol, test_incontrol, test_outcontrol : pd.DataFrame
    """
    train_ic = train[train[0] == incontrol_label].iloc[:, 1:]
    test_ic  = test[test[0] == incontrol_label].iloc[:, 1:]
    test_oc  = test[test[0] != incontrol_label].iloc[:, 1:]

    combined = pd.concat([train_ic, test_ic], ignore_index=True)

    rng = np.random.default_rng(seed)
    shuffled = combined.sample(frac=1, random_state=rng).reset_index(drop=True)

    half_size = len(shuffled) // 2
    train_incontrol = shuffled.iloc[:half_size].reset_index(drop=True)
    test_incontrol  = shuffled.iloc[half_size:].reset_index(drop=True)

    return train_incontrol, test_incontrol, test_oc


def z_norm_each_series(X):
    """
    Z-normalize each time series (row-wise).

    Parameters
    ----------
    X : array-like, shape (n_series, T)

    Returns
    -------
    X_norm : np.ndarray, shape (n_series, T)
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / sd


def apply_global_shift_segment_mean(X, shift_range=5, rng=None):
    """
    Apply global time shift to each series by k ~ U{-shift_range, ..., +shift_range}.

    For k > 0: shift right and fill the left gap using mean of the dropped segment.
    For k < 0: shift left and fill the right gap using mean of the dropped segment.

    Parameters
    ----------
    X : array-like, shape (n_series, T)
    shift_range : int
        Maximum absolute shift.
    rng : np.random.Generator or None

    Returns
    -------
    X_shifted : np.ndarray, shape (n_series, T)
    shifts    : np.ndarray, shape (n_series,)
    """
    X = np.asarray(X, dtype=float)
    n, T = X.shape

    if rng is None:
        rng = np.random.default_rng(0)

    X_out = np.empty_like(X)
    shifts = np.empty(n, dtype=int)

    for i in range(n):
        ts = X[i]
        k = int(rng.integers(-shift_range, shift_range + 1))
        shifts[i] = k

        if k == 0:
            X_out[i] = ts
            continue

        ts_new = np.empty_like(ts)

        if k > 0:
            ts_new[k:] = ts[:-k]
            fill_val = float(np.mean(ts[:k])) if k <= T else float(np.mean(ts))
            ts_new[:k] = fill_val
        else:
            k_abs = -k
            ts_new[:T-k_abs] = ts[k_abs:]
            fill_val = float(np.mean(ts[-k_abs:])) if k_abs <= T else float(np.mean(ts))
            ts_new[T-k_abs:] = fill_val

        X_out[i] = ts_new

    return X_out, shifts