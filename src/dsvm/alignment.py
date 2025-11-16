import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw

def shift_series_time(ts, shift: int):
    """
    시계열 ts를 시간축 방향으로 평행이동.

    Parameters
    ----------
    ts : array-like, shape (T,)
        원본 시계열
    shift : int
        > 0 이면 오른쪽으로, < 0 이면 왼쪽으로 이동

    Returns
    -------
    ts_new : np.ndarray, shape (T,)
    """
    ts = np.asarray(ts, dtype=float)
    ts_new = np.empty_like(ts)

    if shift == 0:
        ts_new[:] = ts
    elif shift > 0:
        ts_new[shift:] = ts[:-shift]
        fill_val = ts[:shift].mean() if shift <= len(ts) else ts.mean()
        ts_new[:shift] = fill_val
    else:
        k_abs = -shift
        ts_new[:len(ts) - k_abs] = ts[k_abs:]
        fill_val = ts[-k_abs:].mean() if k_abs <= len(ts) else ts.mean()
        ts_new[len(ts) - k_abs:] = fill_val

    return ts_new


def compute_dtw_distance_and_path(ts1, ts2, window=None):
    """
    두 시계열 사이의 DTW 거리와 warping path 계산.
    """
    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)

    if window is not None:
        dist = dtw.distance_fast(ts1, ts2, window=window)
        path = dtw.warping_path(ts1, ts2, window=window)
    else:
        dist = dtw.distance_fast(ts1, ts2)
        path = dtw.warping_path(ts1, ts2)

    return dist, path


def compute_euclidean_alignment(ts1, ts2):
    """
    유클리디안 정렬: 동일 인덱스끼리만 매칭하는 정렬과 비용 계산.
    """
    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)
    min_len = min(len(ts1), len(ts2))

    pairs = [(i, i) for i in range(min_len)]
    cost = np.linalg.norm(ts1[:min_len] - ts2[:min_len])

    return pairs, cost


def plot_alignment(x, y, pairs, label_x="Original", label_y="Shifted", title=None, max_lines=400, offset=None):
    """
    두 시계열과 매칭 쌍(pairs)을 이용해 정렬 관계 시각화.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    m = len(y)

    if offset is None:
        rng = max(x.max() - x.min(), y.max() - y.min())
        offset = rng if rng > 0 else 5.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.arange(n), x, lw=2, label=label_x)
    ax.plot(np.arange(m), y - offset, lw=2, label=label_y)

    if len(pairs) > max_lines:
        idx = np.linspace(0, len(pairs) - 1, max_lines).astype(int)
        pairs_to_draw = [pairs[k] for k in idx]
    else:
        pairs_to_draw = pairs

    for i, j in pairs_to_draw:
        if 0 <= i < n and 0 <= j < m:
            ax.plot([i, j], [x[i], y[j] - offset], linewidth=0.6, alpha=0.6)

    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_euclidean_and_dtw_alignment(ts, shift: int, window: int | None = None, max_lines_euclid=300, max_lines_dtw=800):
    """
    하나의 시계열 ts와, 그것을 shift만큼 평행이동한 시계열을 가지고
    (a) 유클리디안 정렬, (b) DTW 정렬 둘 다 시각화.

    Parameters
    ----------
    ts : array-like, shape (T,)
        원본 시계열
    shift : int
        시간 평행이동 크기
    window : int or None
        DTW Sakoe-Chiba window. 없으면 full DTW.
    """
    ts = np.asarray(ts, dtype=float)
    ts_shifted = shift_series_time(ts, shift)

    # Euclidean alignment
    pairs_euclid, euclid_cost = compute_euclidean_alignment(ts, ts_shifted)
    print(f"Euclidean cost: {euclid_cost:.4f}")

    plot_alignment(
        ts,
        ts_shifted,
        pairs_euclid,
        label_x="Original",
        label_y=f"Shifted ({shift:+d})",
        title="(a) Euclidean alignment",
        max_lines=max_lines_euclid,
    )

    # DTW alignment
    dtw_cost, path = compute_dtw_distance_and_path(ts, ts_shifted, window=window)
    print(f"DTW cost: {dtw_cost:.4f}")

    plot_alignment(
        ts,
        ts_shifted,
        path,
        label_x="Original",
        label_y=f"Shifted ({shift:+d})",
        title="(b) DTW alignment",
        max_lines=max_lines_dtw,
    )