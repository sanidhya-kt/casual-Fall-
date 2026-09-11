import numpy as np
from typing import Optional, Tuple

class ChannelNormalizer:
    """
    Standard Z-score normalizer per channel: (x - mean) / std.
    
    [NOT SPECIFIED IN PAPER]: Normalization details were omitted in the text.
    [REPLICATION ASSUMPTION]: Channel-wise standardization based on training set statistics.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted = False
        
    def fit(self, data: np.ndarray) -> "ChannelNormalizer":
        """
        Computes channel-wise mean and standard deviation.
        
        Args:
            data: np.ndarray of shape (N, C, delta) or (Total_samples, C)
        """
        if data.ndim == 3:
            # Shape is (N, C, delta) -> compute mean across (N, delta) per channel
            self.mean = np.mean(data, axis=(0, 2), keepdims=True)  # (1, C, 1)
            self.std = np.std(data, axis=(0, 2), keepdims=True)    # (1, C, 1)
        elif data.ndim == 2:
            self.mean = np.mean(data, axis=0, keepdims=True)
            self.std = np.std(data, axis=0, keepdims=True)
        else:
            raise ValueError(f"Unsupported data ndim: {data.ndim}")
            
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self.is_fitted = True
        return self
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Applies z-score standardization using fitted statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("ChannelNormalizer must be fitted before calling transform.")
        return (data - self.mean) / self.std

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)
