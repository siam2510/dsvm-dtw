import numpy as np

def estimate_arl_for_h_mcc(h, p_swk_mat, m=100):
    """
    MCC(Monte Carlo with Censoring) 방식으로
    주어진 control limit h에 대한 ARL, SDRL, SE를 추정.

    Parameters
    ----------
    h : float
        control limit (threshold)
    p_swk_mat : np.ndarray
        shape = (m, outer_reps) 인 행렬. 각 열이 한 번의 시뮬레이션 스트림.
    m : int
        한 스트림의 최대 길이(우측 검열 시점)

    Returns
    -------
    results : dict
        {
          'arl':    ARL 추정치 (ÂRL_mcc),
          'sdrl':   SDRL 추정치 (표준편차),
          'se':     ARL 표준오차 추정치 (sdrl / sqrt(N)),
          'N':      총 반복 횟수,
          'N0':     uncensored 개수,
          'N_plus': censored 개수
        }
    """
    import numpy as np

    outer_reps = p_swk_mat.shape[1]
    Y = []      # 각 스트림의 '관측된' run length
    delta = []  # 0=uncensored(신호 발생), 1=censored(끝까지 무신호)

    for r in range(outer_reps):
        exceed_idx = np.where(p_swk_mat[:, r] > h)[0]

        if len(exceed_idx) > 0:
            rl = exceed_idx[0] + 1  # 관측된 run length (0-index → 1-index 보정)
            delta.append(0)
        else:
            rl = m                  # 우측 검열
            delta.append(1)

        Y.append(rl)

    Y = np.array(Y, dtype=float)
    delta = np.array(delta, dtype=int)

    N = outer_reps
    N0 = np.sum(delta == 0)     # uncensored
    N_plus = np.sum(delta == 1) # censored

    # uncensored가 하나도 없으면, ARL을 m으로만 둘 수 있고 분산/SE는 추정 불가
    if N0 == 0:
        return {
            'arl': float(m),
            'sdrl': np.nan,
            'se': np.nan,
            'N': int(N),
            'N0': int(N0),
            'N_plus': int(N_plus),
        }

    # ÂRL_mcc = Ȳ_obs + (N_plus/N0) * m
    Y_obs = Y[delta == 0]
    Ybar_obs = Y_obs.mean()
    arl_hat = Ybar_obs + (N_plus / N0) * m

    # μ̂2 = Ȳ2_obs + (N_plus/N0) * (m^2 + 2 m * ÂRL_mcc)
    Y2bar_obs = (Y_obs**2).mean()
    mu2_hat = Y2bar_obs + (N_plus / N0) * (m**2 + 2*m*arl_hat)

    # SDRL = sqrt( μ̂2 - (ÂRL_mcc)^2 )
    sdrl_hat = float(np.sqrt(max(mu2_hat - arl_hat**2, 0.0)))

    # Standard Error of ARL estimator ~ SDRL / sqrt(N)
    se_hat = float(sdrl_hat / np.sqrt(N))

    return {
        'arl': float(arl_hat),
        'sdrl': sdrl_hat,
        'se': se_hat,
        'N': int(N),
        'N0': int(N0),
        'N_plus': int(N_plus),
    }
