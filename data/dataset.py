import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
from preprocessing.normalizer import ChannelNormalizer

class FallDataset(Dataset):
    """
    PyTorch Dataset for Fall Prediction.
    Each sample X has shape (C, delta).
    Label y is integer 0 (ADL) or 1 (Fall).
    """
    def __init__(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        impact_data: Optional[np.ndarray] = None
    ):
        """
        Args:
            X_data: np.ndarray of shape (N, C, delta)
            y_data: np.ndarray of shape (N,)
            impact_data: Optional np.ndarray of shape (N, C, delta) for CEE precomputation
        """
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)
        self.impact = torch.tensor(impact_data, dtype=torch.float32) if impact_data is not None else None
        
    def __len__(self) -> int:
        return len(self.y)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_train_test_split(
    X_list: List[np.ndarray],
    y_list: List[int],
    impact_list: Optional[List[Optional[np.ndarray]]] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    normalize: bool = True
) -> Tuple[FallDataset, FallDataset, Optional[ChannelNormalizer], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Randomly divides dataset into training and testing splits.
    [PAPER SPECIFICATION] Section 5.3: "in a manner that randomly divides datasets"
    [REPLICATION ASSUMPTION]: 80% train, 20% test, stratified by class.
    
    Returns:
        train_dataset, test_dataset, normalizer, train_impact_falls, train_falling_falls
    """
    np.random.seed(random_seed)
    
    X_arr = np.array(X_list, dtype=np.float32)  # (N, C, delta)
    y_arr = np.array(y_list, dtype=np.int64)    # (N,)
    
    # Stratified split indices
    fall_indices = np.where(y_arr == 1)[0]
    adl_indices = np.where(y_arr == 0)[0]
    
    np.random.shuffle(fall_indices)
    np.random.shuffle(adl_indices)
    
    n_train_fall = int(len(fall_indices) * train_ratio)
    n_train_adl = int(len(adl_indices) * train_ratio)
    
    train_idx = np.concatenate([fall_indices[:n_train_fall], adl_indices[:n_train_adl]])
    test_idx = np.concatenate([fall_indices[n_train_fall:], adl_indices[n_train_adl:]])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    X_train = X_arr[train_idx]
    y_train = y_arr[train_idx]
    X_test = X_arr[test_idx]
    y_test = y_arr[test_idx]
    
    normalizer = None
    if normalize:
        normalizer = ChannelNormalizer()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)
        
    train_dataset = FallDataset(X_train, y_train)
    test_dataset = FallDataset(X_test, y_test)
    
    # Extract training fall samples (L_falling and L_impact) for offline Granger causality
    train_fall_mask = (y_train == 1)
    train_falling_falls = X_train[train_fall_mask]
    
    train_impact_falls = None
    if impact_list is not None:
        impact_falls_list = []
        for idx in train_idx:
            if y_arr[idx] == 1 and impact_list[idx] is not None:
                impact_falls_list.append(impact_list[idx])
        if impact_falls_list:
            train_impact_falls = np.array(impact_falls_list, dtype=np.float32)
            if normalize and normalizer is not None:
                train_impact_falls = normalizer.transform(train_impact_falls)
                
    return train_dataset, test_dataset, normalizer, train_impact_falls, train_falling_falls


def load_preprocessed_dataset(
    cache_path: str,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    normalize: bool = True
) -> Tuple[FallDataset, FallDataset, Optional[ChannelNormalizer], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads preprocessed dataset from an NPZ file (created by build_cache.py).
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Preprocessed cache not found at {cache_path}")
        
    data = np.load(cache_path, allow_pickle=True)
    X = data['X']        # (N, C, delta)
    y = data['y']        # (N,)
    impact = data['impact']  # (N, C, delta)
    
    impact_list = [impact[i] if y[i] == 1 else None for i in range(len(y))]
    
    return create_train_test_split(
        X, y, impact_list,
        train_ratio=train_ratio,
        random_seed=random_seed,
        normalize=normalize
    )
