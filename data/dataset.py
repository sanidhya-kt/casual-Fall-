import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
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


def _subject_id(meta: str) -> str:
    match = re.search(r"(?:SE|SA|S)(\d+)", str(meta).upper())
    return match.group(0) if match else str(meta)


def create_train_val_test_split(
    X_list: List[np.ndarray],
    y_list: List[int],
    impact_list: Optional[List[Optional[np.ndarray]]] = None,
    meta_list: Optional[List[str]] = None,
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    random_seed: int = 42,
    normalize: bool = True,
    split_mode: str = "file",
) -> Tuple[FallDataset, FallDataset, FallDataset, Optional[ChannelNormalizer], Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    """Create leakage-aware train/validation/test splits.

    ``file`` is the primary paper-supported interpretation of a random dataset
    split: all windows from one original recording stay together. ``window``
    reproduces the historical implementation. ``subject`` keeps subject IDs
    together and is provided as a diagnostic protocol.
    """
    if split_mode not in {"file", "window", "subject"}:
        raise ValueError("split_mode must be 'file', 'window', or 'subject'")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave a non-empty test split")

    X_arr = np.asarray(X_list, dtype=np.float32)
    y_arr = np.asarray(y_list, dtype=np.int64)
    n_samples = len(y_arr)
    if meta_list is None:
        meta_list = [f"window_{i}" for i in range(n_samples)]
    if len(meta_list) != n_samples:
        raise ValueError("meta_list must have one entry per window")
    if impact_list is None:
        impact_list = [None] * n_samples

    groups = np.asarray(meta_list, dtype=str)
    if split_mode == "subject":
        groups = np.asarray([_subject_id(value) for value in meta_list], dtype=str)
    elif split_mode == "window":
        groups = np.asarray([f"window_{i}" for i in range(n_samples)], dtype=str)

    unique_groups = np.unique(groups)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_groups)
    n_train_groups = max(1, int(len(unique_groups) * train_ratio))
    n_val_groups = max(1, int(len(unique_groups) * val_ratio))
    if n_train_groups + n_val_groups >= len(unique_groups):
        n_val_groups = max(1, len(unique_groups) - n_train_groups - 1)
    train_groups = set(unique_groups[:n_train_groups])
    val_groups = set(unique_groups[n_train_groups:n_train_groups + n_val_groups])
    test_groups = set(unique_groups[n_train_groups + n_val_groups:])

    train_idx = np.where(np.isin(groups, list(train_groups)))[0]
    val_idx = np.where(np.isin(groups, list(val_groups)))[0]
    test_idx = np.where(np.isin(groups, list(test_groups)))[0]
    for indices in (train_idx, val_idx, test_idx):
        rng.shuffle(indices)

    normalizer = ChannelNormalizer() if normalize else None
    X_train = X_arr[train_idx]
    X_val = X_arr[val_idx]
    X_test = X_arr[test_idx]
    if normalizer is not None:
        X_train = normalizer.fit_transform(X_train)
        X_val = normalizer.transform(X_val)
        X_test = normalizer.transform(X_test)

    train_dataset = FallDataset(X_train, y_arr[train_idx])
    val_dataset = FallDataset(X_val, y_arr[val_idx])
    test_dataset = FallDataset(X_test, y_arr[test_idx])

    train_fall_mask = y_arr[train_idx] == 1
    train_falling_falls = X_train[train_fall_mask]
    train_impact_falls = [impact_list[i] for i in train_idx if y_arr[i] == 1 and impact_list[i] is not None]
    train_impact_falls_arr = np.asarray(train_impact_falls, dtype=np.float32) if train_impact_falls else None
    if train_impact_falls_arr is not None and normalizer is not None:
        train_impact_falls_arr = normalizer.transform(train_impact_falls_arr)

    def count(indices: np.ndarray) -> Dict[str, int]:
        return {"total": int(len(indices)), "falls": int(np.sum(y_arr[indices] == 1)), "adls": int(np.sum(y_arr[indices] == 0))}

    overlap = {
        "train_val_groups": len(train_groups & val_groups),
        "train_test_groups": len(train_groups & test_groups),
        "val_test_groups": len(val_groups & test_groups),
    }
    metadata = {
        "split_mode": split_mode,
        "source_files": int(len(set(map(str, meta_list)))),
        "source_subjects": int(len(set(_subject_id(value) for value in meta_list))),
        "train": count(train_idx),
        "validation": count(val_idx),
        "test": count(test_idx),
        "group_overlap": overlap,
        "train_groups": sorted(map(str, train_groups)),
        "validation_groups": sorted(map(str, val_groups)),
        "test_groups": sorted(map(str, test_groups)),
    }
    return (train_dataset, val_dataset, test_dataset, normalizer,
            train_impact_falls_arr, train_falling_falls, metadata)


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
