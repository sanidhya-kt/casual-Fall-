import numpy as np
from scipy import signal

def resample_signal(data: np.ndarray, orig_sr: int, target_sr: int = 25) -> np.ndarray:
    """
    Resamples multi-channel IMU time series from orig_sr to target_sr.
    [PAPER SPECIFICATION] Section 5.2.3:
        "We fixed the sampling frequency to 25Hz for all the datasets."
        SisFall: 200Hz -> 25Hz (factor 8)
        KFall: 100Hz -> 25Hz (factor 4)
    
    Args:
        data: np.ndarray of shape (T, C) or (C, T)
        orig_sr: original sampling rate in Hz (e.g. 200 or 100)
        target_sr: target sampling rate in Hz (default: 25)
        
    Returns:
        resampled_data: np.ndarray of shape (T_new, C)
    """
    if orig_sr == target_sr:
        return data.copy()
        
    was_transposed = False
    if data.ndim == 2 and data.shape[0] < data.shape[1] and data.shape[0] in [6, 9]:
        # Shape is (C, T), transpose to (T, C)
        data = data.T
        was_transposed = True

    # Use resample_poly for accurate, anti-aliased resampling using integer ratios
    # e.g., 25 / 200 = 1 / 8; 25 / 100 = 1 / 4
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    
    resampled = signal.resample_poly(data, up, down, axis=0)
    
    if was_transposed:
        return resampled.T
    return resampled
