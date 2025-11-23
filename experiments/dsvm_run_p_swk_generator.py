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
from multiprocessing import Pool
from tqdm import tqdm

from dsvm.distances import (
    compute_euclidean_distance_matrix,
    compute_dtw_distance_matrix,
)
from dsvm.dsvm_chart import single_rep_run

# ------------------------- 설정 -------------------------

mode = "ecd"                # "ecd", "dtw"
stream = "test_outcontrol"  # "train_incontrol", "test_incontrol", "test_outcontrol"
data_version = "shift20"   # "original", "shift20", "shift40"
random_seed = 10

window = 20                 # window size for DTW, mode이 "dtw"일 때만 사용.
outer_reps = 1000             # 반복 실험 횟수 (병렬 처리로 실행됨)
ref_sample_size = 100       # 참조 집합 크기 (N_0)
N_w = 20                    # 슬라이딩 윈도우 크기
m = 100                     # max length

# 청크 모드 설정
chunk_mode = False      # 청크로 쪼개서 돌릴지 여부
start_rep = 800            # 이번 청크 시작 rep (포함)
end_rep = 1000              # 이번 청크 끝 rep (제외, 즉 0 ~ end_rep -1를 의미)

# 논문 방식 bootstrap 사용 여부
# - 분포를 모르는 in-control ARL0 설계용(train data)일 때 True로 두는 걸 권장
use_bootstrap_for_stream = True

# ------------------------- 데이터 로딩 -------------------------

if data_version == "original":
    train_path = DATA_DIR / "train_incontrol_original.csv"
    test_ic_path = DATA_DIR / "test_incontrol_original.csv"
    test_oc_path = DATA_DIR / "test_outcontrol_original.csv"

elif data_version == "shift20":
    train_path = DATA_DIR / "train_incontrol_shift_20.csv"
    test_ic_path = DATA_DIR / "test_incontrol_shift_20.csv"
    test_oc_path = DATA_DIR / "test_outcontrol_shift_20.csv"

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

# ------------------------- ref / pool / stream 기본 구성 -------------------------

# 1) reference set S0: 항상 train_incontrol에서 ref_sample_size개 비복원추출
X_ref_df = train_incontrol.sample(n=ref_sample_size, random_state=random_seed)
X_ref = X_ref_df.values
y_ref = -np.ones(len(X_ref))  # reference data의 레이블은 -1 (in-control)

# S_rest: bootstrap 재료가 되는 "remaining N - N0 samples"
S_rest_df = train_incontrol.drop(index=X_ref_df.index)
S_rest = S_rest_df.values

# 기존 방식에서 쓰던 기본 pool/stream도 만들어 둔다.
# - bootstrap을 쓰지 않을 때는 이 기본값이 그대로 사용된다.
excluded_indices = X_ref_df.index

if stream == "train_incontrol":
    X_pool_df_base = train_incontrol.drop(index=excluded_indices).sample(
        n=100, random_state=random_seed) # stream 초반에는 buffer가 충분히 차지 않기 때문에, 윈도우를 구성할 때 이 pool을 사용해서 정상 데이터로 앞부분을 채움
    X_pool_base = X_pool_df_base.values
    excluded_indices_all = X_ref_df.index.union(X_pool_df_base.index)
    X_stream_base = train_incontrol.drop(index=excluded_indices_all).values # train_incontrol에서 ref/pool 제외한 나머지를 스트림으로 사용

elif stream == "test_incontrol":
    X_pool_df_base = train_incontrol.drop(index=excluded_indices)
    X_pool_base = X_pool_df_base.values
    X_stream_base = test_incontrol.values

elif stream == "test_outcontrol":
    X_pool_df_base = train_incontrol.drop(index=excluded_indices)
    X_pool_base = X_pool_df_base.values
    X_stream_base = test_outcontrol.values

else:
    raise ValueError("stream은 'train_incontrol', 'test_incontrol', 'test_outcontrol' 중 하나여야 합니다.")

print(f"Stream: {stream}")
print("X_ref      :", X_ref.shape)
print("S_rest     :", S_rest.shape)
print("X_pool_base:", X_pool_base.shape)
print("X_stream_base:", X_stream_base.shape)

# ------------------------- Sigma^2 (RBF 커널 대역폭) 자동 계산 -------------------------
# RBF(Radial Basis Function) 커널의 대역폭 파라미터 sigma^2를 자동으로 추정

if mode == "ecd":
    dists = compute_euclidean_distance_matrix(X_ref)

elif mode == "dtw":
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

def worker(rep: int):
    """
    각 rep마다
    - bootstrap을 쓰는 경우: S_rest에서 복원추출해 X_pool_rep, X_stream_rep 구성
    - bootstrap을 쓰지 않는 경우: X_pool_base, X_stream_base 그대로 사용
    을 한 뒤, single_rep_run으로 p_swk(t) 벡터를 반환한다.
    """
    # rep마다 독립적인 RNG 생성 (multiprocessing에서도 결정적 재현 가능)
    rng = np.random.default_rng(random_seed + rep)

    # 기본값은 기존 코드와 동일한 pool/stream
    X_pool = X_pool_base
    X_stream_raw = X_stream_base

    # 논문식 bootstrap: in-control 분포를 모를 때,
    # remaining N - N0 samples(S_rest)에서 복원추출하여
    # "가짜 in-control stream"을 매 rep마다 새로 생성
    if use_bootstrap_for_stream and stream == "train_incontrol" :
        # pool 크기는 기존 pool과 동일하게 유지
        pool_size = X_pool_base.shape[0]
        # stream 길이는 기존 stream과 동일한 샘플 수로 맞춰줌
        stream_size = X_stream_base.shape[0]

        # S_rest에서 복원추출(with replacement)
        pool_indices = rng.integers(0, S_rest.shape[0], size=pool_size)
        stream_indices = rng.integers(0, S_rest.shape[0], size=stream_size)

        X_pool = S_rest[pool_indices]
        X_stream_raw = S_rest[stream_indices]

    # single_rep_run은 기존 인터페이스 그대로 사용
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

    # 1) 이번에 돌릴 rep 범위 결정
    if chunk_mode:
        # 예: start_rep=0, end_rep=500이면 0~499
        rep_range = range(start_rep, end_rep)   # [start_rep, end_rep)
    else:
        # 기존처럼 0 ~ outer_reps-1 전체
        rep_range = range(outer_reps)

    print(f"실행 rep 범위: {rep_range.start} ~ {rep_range.stop - 1}")
    print(f"총 {len(rep_range)}개 rep 실행")

    # 2) 병렬 실행
    all_pswk = []
    with Pool(processes=10) as pool:
        for res in tqdm(pool.imap(worker, rep_range), total=len(rep_range)):
            all_pswk.append(res)

    # all_pswk: 리스트 길이 = len(rep_range), 각 원소는 길이 m 벡터
    p_swk_mat = np.array(all_pswk).T  # shape: (m, len(rep_range))

    # 컬럼 이름도 현재 rep 번호에 맞게
    df_pswk = pd.DataFrame(
        p_swk_mat,
        columns=[f"rep_{r}" for r in rep_range],
    )
    df_pswk.index.name = "t"

    save_dir = PROJECT_ROOT / "results" / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3) 파일 이름에 이번 청크의 rep 범위 반영
    if chunk_mode:
        rep_str = f"{rep_range.start}~{rep_range.stop - 1}"   # 예: "0~499"
    else:
        rep_str = f"0~{len(rep_range)-1}"

    save_name = (
        f"ECG5000_{stream}_p_swk_matrix_{mode}_{data_version}"
        f"({rep_str})_bootstrap_seed{random_seed}_m={m}.csv"
    )
    save_path = save_dir / save_name

    df_pswk.to_csv(save_path)
    print(f"저장 완료: {save_path}")
