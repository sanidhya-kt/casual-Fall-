import numpy as np
from typing import Tuple, Optional, List
from preprocessing.impact_detector import detect_impact_time

def extract_fall_windows(
    data: np.ndarray,
    acc_indices: Tuple[int, int, int] = (0, 1, 2),
    delta: int = 72,
    theta: int = 15,
    extract_impact: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """
    Extracts the Falling window L_falling and optionally the Impact window L_impact from a fall trial.
    
    [PAPER SPECIFICATION] Section 3, Phase 2:
        L_falling = [t_impact - theta - delta, t_impact - theta)
        Input to model X = {x_1^c, ..., x_delta^c}
        delta = 72 (from Sec 5.2.3 experimental setup), theta = 15
        
    [PAPER SPECIFICATION] Section 4.2 (CEE):
        L_impact = [t_impact, t_impact + delta)
        
    Args:
        data: np.ndarray of shape (T, C) containing 25Hz resampled IMU data.
        acc_indices: tuple of 3 channel indices for accelerometer (default (0,1,2)).
        delta: window length (default 72).
        theta: lead time / protection horizon (default 15).
        extract_impact: whether to extract L_impact for CEE Granger causality.
        
    Returns:
        x_falling: np.ndarray of shape (C, delta)
        y_impact: np.ndarray of shape (C, delta) or None
        t_impact: detected impact index
    """
    if data.shape[0] < data.shape[1] and data.shape[0] in [6, 9]:
        data = data.T  # Ensure (T, C)
        
    acc_data = data[:, list(acc_indices)]
    t_impact = detect_impact_time(acc_data)
    
    start_fall = t_impact - theta - delta
    end_fall = t_impact - theta
    
    # Boundary handling if fall starts too early
    if start_fall < 0:
        # [REPLICATION ASSUMPTION]: If recording begins too close to impact,
        # zero-pad at the beginning to maintain fixed window length delta.
        pad_len = -start_fall
        segment = data[0:max(0, end_fall), :]
        x_falling = np.pad(segment, ((pad_len, 0), (0, 0)), mode='constant')
    else:
        x_falling = data[start_fall:end_fall, :]
        
    # Ensure exact length delta
    if x_falling.shape[0] != delta:
        x_falling = np.pad(x_falling, ((0, max(0, delta - x_falling.shape[0])), (0, 0)), mode='constant')[:delta, :]
        
    y_impact = None
    if extract_impact:
        start_impact = t_impact
        end_impact = t_impact + delta
        if end_impact > data.shape[0]:
            pad_len = end_impact - data.shape[0]
            segment = data[start_impact:, :]
            y_impact = np.pad(segment, ((0, pad_len), (0, 0)), mode='edge')
        else:
            y_impact = data[start_impact:end_impact, :]
            
        if y_impact.shape[0] != delta:
            y_impact = np.pad(y_impact, ((0, max(0, delta - y_impact.shape[0])), (0, 0)), mode='edge')[:delta, :]
            
        # Shape to (C, delta)
        y_impact = y_impact.T
        
    # Transpose x_falling to (C, delta) to match [PAPER SPECIFICATION] B x C x delta
    x_falling = x_falling.T
    
    return x_falling, y_impact, t_impact


def extract_adl_windows(
    data: np.ndarray,
    acc_indices: Tuple[int, int, int] = (0, 1, 2),
    delta: int = 72,
    theta: int = 15,
    mode: str = 'peak'
) -> List[np.ndarray]:
    """
    Extracts window(s) of length delta from an ADL (non-fall) trial.
    
    [NOT SPECIFIED IN PAPER]: Exact sampling mechanism for ADL sequences.
    [REPLICATION ASSUMPTION]:
        - 'peak': Extract window around peak SMV (mimicking fall window extraction relative to peak acceleration).
        - 'sliding': Extract non-overlapping sliding windows across the ADL sequence.
    
    Args:
        data: np.ndarray of shape (T, C)
        acc_indices: tuple of 3 channel indices for accelerometer
        delta: window length (default 72)
        theta: lead time (default 15)
        mode: 'peak' or 'sliding'
        
    Returns:
        List of np.ndarray windows of shape (C, delta)
    """
    if data.shape[0] < data.shape[1] and data.shape[0] in [6, 9]:
        data = data.T
        
    T, C = data.shape
    windows = []
    
    if mode == 'peak':
        acc_data = data[:, list(acc_indices)]
        t_peak = detect_impact_time(acc_data)
        start = t_peak - theta - delta
        end = t_peak - theta
        
        if start < 0:
            pad_len = -start
            segment = data[0:max(0, end), :]
            w = np.pad(segment, ((pad_len, 0), (0, 0)), mode='constant')
        elif end > T:
            w = data[max(0, T - delta):T, :]
            if w.shape[0] < delta:
                w = np.pad(w, ((delta - w.shape[0], 0), (0, 0)), mode='constant')
        else:
            w = data[start:end, :]
            
        if w.shape[0] != delta:
            w = np.pad(w, ((0, max(0, delta - w.shape[0])), (0, 0)), mode='constant')[:delta, :]
            
        windows.append(w.T)  # (C, delta)
        
    elif mode == 'sliding':
        stride = delta // 2  # 50% overlap or non-overlapping
        for start in range(0, max(1, T - delta + 1), stride):
            w = data[start:start + delta, :]
            if w.shape[0] == delta:
                windows.append(w.T)
        if not windows:
            # If sequence is shorter than delta, pad it
            w = np.pad(data, ((0, delta - T), (0, 0)), mode='constant')
            windows.append(w.T)
            
    return windows
