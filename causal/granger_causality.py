import numpy as np
from typing import Tuple, List, Optional

def fit_linear_regression(X_mat: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
    """
    Fits Ordinary Least Squares (OLS): beta = (X^T X)^(-1) X^T y using pseudo-inverse for stability.
    """
    beta, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    return beta


def compute_bivariate_granger_causality(
    x_cause: np.ndarray,
    y_effect: np.ndarray,
    max_lag: int = 5
) -> float:
    """
    Calculates the Granger Causality metric M_GC(x -> y) as defined in [PAPER SPECIFICATION] Eq. (5)-(9):
        Restricted VAR: y_t = sum_{j=1}^q alpha_j * y_{t-j} + eps_y
        Unrestricted VAR: y_t = sum_{k=1}^r alpha_k * y_{t-k} + sum_{l=1}^s beta_l * x_{t-l} + eps_xy
        M_GC = ln( var(eps_y) / var(eps_xy) )
    
    Args:
        x_cause: 1D np.ndarray of length delta (from L_falling)
        y_effect: 1D np.ndarray of length delta (from L_impact)
        max_lag: lag order <= theta (default 5, <= 15)
        
    Returns:
        M_GC: float (log variance ratio)
    """
    T = len(y_effect)
    if T <= max_lag + 1:
        return 0.0
        
    # Build lagged matrices for t = max_lag ... T - 1
    # Target is y_effect[max_lag:]
    y_target = y_effect[max_lag:]
    N = len(y_target)
    
    # Restricted design matrix: lags of y alone + intercept
    X_restr = np.ones((N, max_lag + 1), dtype=np.float64)
    for lag in range(1, max_lag + 1):
        X_restr[:, lag] = y_effect[max_lag - lag : T - lag]
        
    # Unrestricted design matrix: lags of y + lags of x + intercept
    X_unrestr = np.ones((N, 2 * max_lag + 1), dtype=np.float64)
    for lag in range(1, max_lag + 1):
        X_unrestr[:, lag] = y_effect[max_lag - lag : T - lag]
        X_unrestr[:, max_lag + lag] = x_cause[max_lag - lag : T - lag]
        
    # Fit restricted model
    beta_restr = fit_linear_regression(X_restr, y_target)
    eps_y = y_target - X_restr @ beta_restr
    var_restr = np.var(eps_y, ddof=1)
    
    # Fit unrestricted model
    beta_unrestr = fit_linear_regression(X_unrestr, y_target)
    eps_xy = y_target - X_unrestr @ beta_unrestr
    var_unrestr = np.var(eps_xy, ddof=1)
    
    if var_restr <= 1e-12 or var_unrestr <= 1e-12:
        return 0.0
        
    # M_GC [PAPER SPECIFICATION] Eq. (9)
    ratio = var_restr / var_unrestr
    if ratio <= 0:
        return 0.0
        
    m_gc = float(np.log(ratio))
    return max(0.0, m_gc)


def compute_causal_matrix(
    falling_samples: np.ndarray,
    impact_samples: np.ndarray,
    max_lag: int = 5,
    threshold: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the channel-to-channel causal effect matrix and channel weights across fall samples.
    [PAPER SPECIFICATION] Section 4.2 & Eq. (10)-(11):
        E(X^i, Y^j) = M_GC(X^i, Y^j) if M_GC > 1.0 else 0
        w_i = sum_j E(i, j) / sum_{j, k} E(j, k)
        
    Args:
        falling_samples: np.ndarray of shape (N_falls, C, delta)
        impact_samples: np.ndarray of shape (N_falls, C, delta)
        max_lag: lag order <= theta (default: 5)
        threshold: 1.0 as defined in Eq. (10)
        
    Returns:
        E_matrix: np.ndarray of shape (C, C) containing directed causal effect from falling to impact
        channel_weights: np.ndarray of shape (C,) normalized causal weights w_i
    """
    N_falls, C, delta = falling_samples.shape
    E_accum = np.zeros((C, C), dtype=np.float64)
    
    for n in range(N_falls):
        fall_sample = falling_samples[n]    # (C, delta)
        impact_sample = impact_samples[n]  # (C, delta)
        
        for i in range(C):
            for j in range(C):
                m_gc = compute_bivariate_granger_causality(
                    fall_sample[i],
                    impact_sample[j],
                    max_lag=max_lag
                )
                # Eq. (10) thresholding
                if m_gc > threshold:
                    E_accum[i, j] += m_gc

    # Average over number of fall samples
    if N_falls > 0:
        E_matrix = E_accum / N_falls
    else:
        E_matrix = np.ones((C, C), dtype=np.float64)
        
    # Global denominator: sum_{j=1}^C sum_{k=1}^C E(j, k) [Eq. 11]
    total_causal_sum = np.sum(E_matrix)
    
    if total_causal_sum > 1e-8:
        # w_i = sum_{j=1}^C E(i, j) / total_sum [Eq. 11]
        channel_weights = np.sum(E_matrix, axis=1) / total_causal_sum
    else:
        # Uniform if no significant causality detected
        channel_weights = np.ones(C, dtype=np.float64) / C
        
    return E_matrix, channel_weights
