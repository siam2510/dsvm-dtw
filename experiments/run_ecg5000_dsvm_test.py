# experiments/run_ecg5000_dsvm_s05.py

from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from dsvm.distances import (
    compute_euclidean_distance_matrix,
    compute_dtw_distance_matrix,
)
from dsvm.dsvm_chart import single_rep_run

# ------------------------- 설정 -------------------------

mode = "ecd"              # "ecd" 또는 "dtw"
stream = "test_outcontrol"  # "train_incontrol", "test_incontrol", "test_outcontrol"
data_version = "original"   # "original", "shift20", "shift40"
random_seed = 2025

window = 20
outer_reps = 10
ref_sample_size = 100
N_w = 20
m = 20


# ------------------------- 데이터 로딩 -------------------------

if data_version == "original":
    train_path = DATA_DIR / "train_incontrol_original.csv"
    test_ic_path = DATA_DIR / "test_incontrol_original.csv"
    test_oc_path = DATA_DIR / "test_outcontrol_original.csv"

elif data_version == "shift20":
    train_path = DATA_DIR / "rain_incontrol_shift20.csv"
    test_ic_path = DATA_DIR / "test_incontrol_shift20.csv"
    test_oc_path = DATA_DIR / "test_outcontrol_shift20.csv"

elif data_version == "shift40":
    train_path = DATA_DIR / "train_incontrol_shift_40.csv"
    test_ic_path = DATA_DIR / "test_incontrol_shift_40.csv"
    test_oc_path = DATA_DIR / "test_outcontrol_shift_40.csv"

else:
    raise ValueError("data_version must be 'original', 'shift20', or 'shift40'.")

train_incontrol = pd.read_csv(train_path, header=None)
test_incontrol = pd.read_csv(test_ic_path, header=None)
test_outcontrol = pd.read_csv(test_oc_path, header=None)

print("train_incontrol:", train_incontrol.shape)
print("test_incontrol :", test_incontrol.shape)
print("test_outcontrol:", test_outcontrol.shape)


# ------------------------- ref / pool / stream 구성 -------------------------

X_ref_df = train_incontrol.sample(n=ref_sample_size, random_state=random_seed)
X_ref = X_ref_df.values
y_ref = -np.ones(len(X_ref))

excluded_indices = X_ref_df.index

if stream == "train_incontrol":
    X_pool_df = train_incontrol.drop(index=excluded_indices).sample(
        n=100, random_state=random_seed
    )
    X_pool = X_pool_df.values
    excluded_indices_all = X_ref_df.index.union(X_pool_df.index)
    X_stream_raw = train_incontrol.drop(index=excluded_indices_all).values

elif stream == "test_incontrol":
    X_pool_df = train_incontrol.drop(index=excluded_indices)
    X_pool = X_pool_df.values
    X_stream_raw = test_incontrol.values

elif stream == "test_outcontrol":
    X_pool_df = train_incontrol.drop(index=excluded_indices)
    X_pool = X_pool_df.values
    X_stream_raw = test_outcontrol.values

else:
    raise ValueError("stream은 'train_incontrol', 'test_incontrol', 'test_outcontrol' 중 하나여야 합니다.")

print(f"Stream: {stream}")
print("X_ref   :", X_ref.shape)
print("X_pool  :", X_pool.shape)
print("X_stream:", X_stream_raw.shape)


# ------------------------- sigma^2 계산 -------------------------

if mode == "ecd":
    dists = compute_euclidean_distance_matrix(X_ref)
elif mode == "dtw":
    dists = compute_dtw_distance_matrix(X_ref, window=window)
else:
    raise ValueError("mode는 'ecd' 또는 'dtw'만 가능합니다.")

mask = ~np.eye(len(dists), dtype=bool)
d2_offdiag = (dists ** 2)[mask]
sigma2 = np.median(d2_offdiag)

if not np.isfinite(sigma2) or sigma2 <= 1e-12:
    sigma2 = 1e-12

print(f"자동 sigma^2 (median heuristic, off-diagonal) with {mode}: {sigma2:.6f}")


# ------------------------- worker 함수 -------------------------

def worker(rep):
    return single_rep_run(
        rep=rep,
        X_stream_raw=X_stream_raw,
        X_pool=X_pool,
        X_ref=X_ref,
        y_ref=y_ref,
        sigma2=sigma2,
        N_w=N_w,
        m=m,
        mode=mode,
        window=window,
    )


# ------------------------- 메인 실행 -------------------------

if __name__ == "__main__":
    reps = list(range(outer_reps))
    all_pswk = []

    with Pool(processes=10) as pool:
        for res in tqdm(pool.imap(worker, reps), total=outer_reps):
            all_pswk.append(res)

    p_swk_mat = np.array(all_pswk).T
    df_pswk = pd.DataFrame(
        p_swk_mat,
        columns=[f"rep_{r}" for r in range(outer_reps)],
    )
    df_pswk.index.name = "t"

    save_dir = PROJECT_ROOT / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_name = f"ECG5000_{stream}_p_swk_matrix_{mode}_{data_version}(0~{outer_reps-1})_TEST_seed{random_seed}_m={m}.csv"
    save_path = save_dir / save_name

    df_pswk.to_csv(save_path)
    print(f"저장 완료: {save_path}")
