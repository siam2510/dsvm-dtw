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
    window):
    """
    한 번의 반복 실험(single repetition)에서 시간 스탬프별 이상 탐지 점수를 계산합니다.
    
    이 함수는 스트리밍 데이터 환경에서 Sliding Window + SVM 기반의 이상 탐지를 수행합니다.
    각 시간 스탬프 t마다 새로운 샘플을 받으면서 누적된 윈도우 데이터로 SVM을 훈련하고,
    그 윈도우 내 샘플들에 대한 이상 점수(anomaly score)를 계산합니다.
    
    Parameters
    ----------
    rep : int
        반복 번호 (난수 생성 시드 결정에 사용)
    X_stream_raw : ndarray
        원본 스트림 데이터, shape (n_stream, n_features)
    X_pool : ndarray
        보충 데이터 (초기 버퍼가 부족할 때 윈도우 구성에 사용)
        shape (n_pool, n_features)
    X_ref : ndarray
        참조 집합 (정상 데이터), shape (n_ref, n_features)
    y_ref : ndarray
        참조 집합의 레이블 (보통 -1), length n_ref
    sigma2 : float
        RBF 커널의 대역폭 파라미터
    N_w : int
        슬라이딩 윈도우 크기
    m : int
        처리할 스트림 샘플의 최대 개수
    mode : str
        거리 메트릭 선택 ("ecd" 또는 "dtw")
    window : int or None
        DTW 계산 시 Sakoe-Chiba band 제약 너비
        
    Returns
    -------
    p_swk_list : list
        길이 m의 리스트, 각 요소는 시간 t에서의 이상 점수 (0~1 사이 확률)
    """
    
    # ========================= 초기화 =========================
    # 반복 번호를 시드로 사용하여 재현 가능한 난수 생성기 생성
    # rep + 2000 오프셋은 다른 난수 시드와의 충돌 방지
    rng = np.random.default_rng(rep + 2000)
    
    # 버퍼: 스트림에서 받은 샘플을 누적
    buffer = []
    
    # 스트림 데이터를 무작위로 섞기 (각 반복마다 다른 순서 생성)
    X_stream = rng.permutation(X_stream_raw)
    
    # 각 시간 스탬프에서의 이상 점수를 저장할 리스트
    p_swk_list = []

    # ========================= 시간 스탬프별 처리 루프 =========================
    for t in range(m):
        # --------- 1단계: 새로운 샘플 수신 ---------
        z = X_stream[t]
        # 스트림에서 t번째 샘플을 버퍼에 추가
        buffer.append(z)

        # --------- 2단계: 슬라이딩 윈도우 구성 ---------
        # construct_window_with_backfill 함수:
        #   - N_w 크기의 윈도우 구성
        #   - 버퍼 크기가 N_w보다 작으면 X_pool에서 보충하여 채움
        X_win = construct_window_with_backfill(
            np.array(buffer), t, X_pool, N_w, rng
        )
        # 윈도우 샘플의 레이블 (테스트 샘플 = 1, 참조 샘플 = -1)
        y_win = np.ones(len(X_win))

        # --------- 3단계: 훈련 데이터 구성 ---------
        # 참조 집합과 윈도우 데이터를 합쳐서 훈련 데이터 생성
        # SVM의 결정 경계는 참조(정상) vs 윈도우(테스트) 데이터 사이에 결정
        X_train = np.vstack([X_ref, X_win])
        y_train = np.concatenate([y_ref, y_win])

        # --------- 4단계: 훈련 커널 행렬 계산 ---------
        # 선택한 거리 메트릭에 따라 RBF 커널 계산
        if mode == "ecd":
            # Euclidean Distance 기반 커널
            K_train = custom_ecd_kernel(X_train, X_train, sigma2)
        elif mode == "dtw":
            # DTW 거리 기반 커널
            K_train = custom_dtw_kernel(X_train, X_train, sigma2, window=window)
            # DTW 커널이 완벽한 양정치성을 보장하지 않으므로 PSD 조정
            # psd_shift_min_eig: 가장 작은 고유값을 0으로 시프트하여 양정치성 보장
            K_train, lam_used = psd_shift_min_eig(K_train, eps_abs=1e-12)
        else:
            raise ValueError("mode는 'ecd' 또는 'dtw'만 가능합니다.")

        # --------- 5단계: 커널 정규화 (훈련) ---------
        # 커널 행렬을 정규화하여 모든 데이터 포인트가 동일한 스케일을 가지도록 함
        # d_train: 정규화 계수 (대각 원소들)
        K_train, d_train = normalize_kernel_train(K_train)

        # --------- 6단계: SVM 훈련 ---------
        # 사전 계산된 커널 행렬을 사용하는 SVM
        # C=1.0: 정규화 매개변수 (기본값)
        # kernel="precomputed": 커널 행렬이 직접 제공됨
        clf = SVC(
            kernel="precomputed",
            C=1.0,
            shrinking=False,
            cache_size=1000,
        )
        # X_train 대신 K_train(사전 계산된 커널)을 입력
        clf.fit(K_train, y_train)

        # --------- 7단계: 테스트 커널 행렬 계산 ---------
        # 윈도우 샘플들과 훈련 데이터 간의 커널 계산
        # K_test[i, j] = k(X_win[i], X_train[j])
        if mode == "ecd":
            K_test = custom_ecd_kernel(X_win, X_train, sigma2)
        elif mode == "dtw":
            K_test = custom_dtw_kernel(X_win, X_train, sigma2, window=window)
        else:
            raise ValueError("mode는 'ecd' 또는 'dtw'만 가능합니다.")

        # --------- 8단계: 테스트 커널 정규화 ---------
        # 훈련 시 사용한 정규화 계수(d_train)를 사용하여 테스트 커널 정규화
        d_test = np.ones(len(X_win), dtype=float)
        K_test = normalize_kernel_pair(K_test, d_test, d_train)

        # --------- 9단계: 이상 점수 계산 ---------
        # SVM의 결정 함수(decision function) 계산
        # 참조 데이터에 가까우면 음수, 멀면 양수
        f_scores = clf.decision_function(K_test)
        # print("f_scores : ", f_scores) # test용. 삭제 예정
        # Sigmoid 함수를 통해 점수를 확률로 변환 (0~1 범위)
        # g_scores = 1 / (1 + exp(-f_scores))
        # f_scores >> 0 이면 g_scores ≈ 1 (이상)
        # f_scores << 0 이면 g_scores ≈ 0 (정상)
        g_scores = 1 / (1 + np.exp(-f_scores))
        # print("g_scores : ", g_scores) # test용. 삭제 예정
        # 윈도우 내 모든 샘플의 이상 점수의 평균
        # 이것이 시간 t에서의 최종 이상 탐지 점수
        p_swk = np.mean(g_scores)
        p_swk_list.append(p_swk)
        # print(f"t={t}, p_swk={p_swk:.4f}")

    # ========================= 반환 =========================
    # 각 시간 스탬프 t=0..m-1에서의 이상 점수 리스트 반환
    return p_swk_list
