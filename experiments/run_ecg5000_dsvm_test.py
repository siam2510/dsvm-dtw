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

mode = "ecd"                # "ecd", "dtw"
stream = "test_outcontrol"  # "train_incontrol", "test_incontrol", "test_outcontrol"
data_version = "original"   # "original", "shift20", "shift40"
random_seed = 2025

window = 20                 # window size for DTW, mode이 "dtw"일 때만 사용.
outer_reps = 10             # 반복 실험 횟수 (병렬 처리로 실행됨)
ref_sample_size = 100       # 참조 집합 크기 (N_0)
N_w = 20                    # 슬라이딩 윈도우 크기
m = 100                      # max length

# ------------------------- 데이터 로딩 -------------------------

if data_version == "original":
    train_path = DATA_DIR / "train_incontrol_original.csv"
    test_ic_path = DATA_DIR / "test_incontrol_original.csv"
    test_oc_path = DATA_DIR / "test_outcontrol_original.csv"

elif data_version == "shift20":
    train_path = DATA_DIR / "train_incontrol_shift20.csv"
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

X_ref_df = train_incontrol.sample(n=ref_sample_size, random_state=random_seed) # train_incontrol에서 reference data 샘플링
X_ref = X_ref_df.values 
y_ref = -np.ones(len(X_ref)) # reference data의 레이블은 -1 (in-control)

excluded_indices = X_ref_df.index # reference로 선택된 인덱스는 pool/stream에서 제외

if stream == "train_incontrol":
    X_pool_df = train_incontrol.drop(index=excluded_indices).sample(
        n=100, random_state=random_seed) # stream 초반에는 buffer가 충분히 차지 않기 때문에, 윈도우를 구성할 때 이 pool을 사용해서 정상 데이터로 앞부분을 채움
    X_pool = X_pool_df.values
    excluded_indices_all = X_ref_df.index.union(X_pool_df.index)
    X_stream_raw = train_incontrol.drop(index=excluded_indices_all).values # train_incontrol에서 ref/pool 제외한 나머지를 스트림으로 사용

elif stream == "test_incontrol":
    X_pool_df = train_incontrol.drop(index=excluded_indices)
    X_pool = X_pool_df.values
    X_stream_raw = test_incontrol.values # test_incontrol 전체를 스트림으로 사용

elif stream == "test_outcontrol":
    X_pool_df = train_incontrol.drop(index=excluded_indices)
    X_pool = X_pool_df.values
    X_stream_raw = test_outcontrol.values # test_outcontrol 전체를 스트림으로 사용

else:
    raise ValueError("stream은 'train_incontrol', 'test_incontrol', 'test_outcontrol' 중 하나여야 합니다.")

print(f"Stream: {stream}")
print("X_ref   :", X_ref.shape)
print("X_pool  :", X_pool.shape)
print("X_stream:", X_stream_raw.shape)


# ========================= Sigma^2 (RBF 커널 대역폭) 자동 계산 =========================
# RBF(Radial Basis Function) 커널의 대역폭 파라미터 sigma^2를 자동으로 추정합니다.

if mode == "ecd" :
    dists = compute_euclidean_distance_matrix(X_ref)
    
elif mode == "dtw" : 
    dists = compute_dtw_distance_matrix(X_ref, window=window)
    
else:
    raise ValueError("mode는 'ecd' 또는 'dtw'만 가능합니다.")

# 대각선 원소를 제외하기 위한 마스크 생성
# np.eye()는 항등 행렬(identity matrix)을 생성하고,
# ~를 통해 True/False를 반전시켜 대각선이 False, 나머지가 True인 마스크를 생성
mask = ~np.eye(len(dists), dtype=bool)

# 거리 행렬의 대각선 원소(자기 자신과의 거리 = 0)를 제외한
# 모든 오프대각선(off-diagonal) 원소만 추출하고 제곱
d2_offdiag = (dists ** 2)[mask]

# Median Heuristic: 오프대각선 거리들의 중앙값을 sigma^2로 사용
sigma2 = np.median(d2_offdiag)

# ========================= Sigma^2 안정성 검사 =========================
# 계산된 sigma^2가 유효한지 확인하고, 필요시 최소값으로 설정합니다.

if not np.isfinite(sigma2) or sigma2 <= 1e-12:
    # 만약 sigma^2가 NaN, Inf 또는 매우 작은 값이면 RBF 커널 계산에서 수치적 문제가 발생할 수 있으므로 최소 임계값 1e-12로 설정하여 안정성 확보
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
        window=window)

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
