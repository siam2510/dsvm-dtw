# src/dsvm/distances.py

import numpy as np
from scipy.spatial.distance import cdist
from dtaidistance import dtw


def compute_euclidean_distance_matrix(X):
    """
    입력 데이터 간의 pairwise Euclidean 거리 행렬을 계산합니다.
    
    Parameters
    ----------
    X : array-like
        입력 데이터, shape는 (n_samples, n_features)
        
    Returns
    -------
    dist_mat : ndarray
        (n_samples, n_samples) shape의 거리 행렬
        dist_mat[i, j]는 X[i]와 X[j] 사이의 유클리드 거리
    """
    # 입력을 float 타입의 numpy 배열로 변환하여 계산 안정성 확보
    X = np.asarray(X, dtype=float)
    
    # scipy의 cdist 함수로 pairwise Euclidean 거리 계산
    # metric="euclidean"은 sqrt(sum((x_i - y_i)^2)) 공식 사용
    # cdist는 입력 행렬의 모든 행(samples) 쌍에 대해 거리를 계산
    dist_mat = cdist(X, X, metric="euclidean")
    
    return dist_mat


def compute_dtw_distance_matrix(X, window=None):
    """
    Dynamic Time Warping (DTW)을 이용한 pairwise 거리 행렬을 계산합니다.
    
    Parameters
    ----------
    X : array-like
        입력 시계열 데이터, shape는 (n_samples, n_features)
        각 행은 하나의 시계열 시퀀스를 나타냅니다.
        
    window : int, optional
        Sakoe-Chiba band constraint의 너비
        window가 설정되면 DTW 계산 시 대각선 주변의 범위로 제한
        기본값: None (제약 없음)
        
    Returns
    -------
    dist_mat : ndarray
        (n_samples, n_samples) shape의 대칭 거리 행렬
        dist_mat[i, j]는 X[i]와 X[j] 사이의 DTW 거리
        
    Notes
    -----
    - 대칭성: dist_mat[i, j] == dist_mat[j, i]
    - 상삼각 행렬만 계산하고 대칭성으로 하삼각 채우기
    """
    # 입력을 float 타입의 numpy 배열로 변환
    X = np.asarray(X, dtype=float)
    
    # 샘플 개수 추출
    n = len(X)
    
    # 거리 행렬 초기화 (모두 0으로 시작, 대칭 행렬)
    dist_mat = np.zeros((n, n), dtype=float)

    # 빠른 알고리즘과 느린 알고리즘의 사용 여부를 추적하는 플래그
    # (첫 사용 시에만 메시지 출력하기 위함)
    used_fast = False
    used_slow = False

    # 모든 샘플 쌍의 거리 계산 (상삼각만 계산 후 대칭성 이용)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                # 먼저 fast DTW 계산 시도
                # use_pruning=True: 계산 효율 향상을 위한 가지치기 기법 사용
                # window: Sakoe-Chiba band 제약 (Optional)
                dist = dtw.distance_fast(X[i], X[j], window=window, use_pruning=True)
                if not used_fast:
                    print("Using fast DTW in distance matrix")
                    used_fast = True
            except AttributeError:
                # distance_fast가 없으면 일반 DTW 계산으로 폴백
                # (이전 버전의 dtaidistance 라이브러리 호환성)
                dist = dtw.distance(X[i], X[j], window=window)
                if not used_slow:
                    print("Falling back to slow DTW in distance matrix")
                    used_slow = True
            
            # 계산된 거리를 행렬에 저장 (대칭성 유지)
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    return dist_mat
