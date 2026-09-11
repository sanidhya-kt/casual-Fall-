import os
import glob
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from preprocessing.resampler import resample_signal
from preprocessing.windowing import extract_fall_windows, extract_adl_windows

class KFallLoader:
    """
    Dataset loader for KFall (Yu et al., 2021).
    [PAPER SPECIFICATION] Section 5.1:
        100Hz original sampling frequency.
        32 young subjects.
        21 ADLs types, 15 Fall types, total 5,075 samples.
        Sensors: Accelerometer (3), Gyroscope (3), Orientation (3) at waist = 9 channels.
        Resampled to 25Hz.
    """
    def __init__(
        self,
        root_dir: str,
        target_sr: int = 25,
        orig_sr: int = 100,
        delta: int = 72,
        theta: int = 15
    ):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.orig_sr = orig_sr
        self.delta = delta
        self.theta = theta
        
    def parse_file(self, filepath: str) -> np.ndarray:
        """
        Parses a KFall CSV/text file.
        Extracts 9 channels: Acc (X, Y, Z), Gyro (X, Y, Z), Orientation (X, Y, Z).
        """
        df = pd.read_csv(filepath)
        
        # Identify columns
        # KFall standard headers usually include AccX/Acc_x, GyroX/Gyro_x, EulerX/Roll etc.
        # Or raw columns without header if numeric
        cols = df.columns
        if len(cols) >= 9:
            # Check for header names
            lower_cols = [str(c).lower() for c in cols]
            acc_cols = [c for c in cols if 'acc' in str(c).lower() or 'ax' in str(c).lower() or 'a_x' in str(c).lower()]
            gyro_cols = [c for c in cols if 'gyr' in str(c).lower() or 'gx' in str(c).lower() or 'g_x' in str(c).lower()]
            ori_cols = [c for c in cols if any(k in str(c).lower() for k in ['ori', 'euler', 'angle', 'roll', 'pitch', 'yaw'])]
            
            if len(acc_cols) >= 3 and len(gyro_cols) >= 3 and len(ori_cols) >= 3:
                selected_cols = acc_cols[:3] + gyro_cols[:3] + ori_cols[:3]
                data = df[selected_cols].values.astype(np.float32)
            else:
                # Numerical indexing: columns 1:10 (if col 0 is timestamp) or 0:9
                if pd.api.types.is_numeric_dtype(df[cols[0]]) and np.all(np.diff(df[cols[0]].values[:10]) > 0) and len(cols) > 9:
                    # Column 0 is likely frame/time index
                    data = df.iloc[:, 1:10].values.astype(np.float32)
                else:
                    data = df.iloc[:, :9].values.astype(np.float32)
        else:
            raise ValueError(f"File {filepath} has fewer than 9 columns ({len(cols)})")
            
        return data

    def load_dataset(self) -> Tuple[List[np.ndarray], List[int], List[Optional[np.ndarray]], List[str]]:
        """
        Scans root_dir and loads all available KFall samples.
        
        Returns:
            X_list: list of (C, delta) arrays
            y_list: list of int labels (0 = ADL, 1 = Fall)
            y_impact_list: list of (C, delta) impact arrays for falls (or None for ADLs)
            meta_list: list of file metadata/paths
        """
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"KFall directory not found at {self.root_dir}")
            
        file_patterns = [
            os.path.join(self.root_dir, '**', '*.csv'),
            os.path.join(self.root_dir, '*.csv')
        ]
        files = []
        for p in file_patterns:
            files.extend(glob.glob(p, recursive=True))
        files = sorted(list(set(files)))
        
        X_list = []
        y_list = []
        y_impact_list = []
        meta_list = []
        
        for fpath in files:
            fname = os.path.basename(fpath)
            is_fall = False
            is_adl = False
            
            # Check T<number> task coding in KFall (T01-T21: ADL, T22-T36: Fall)
            import re
            m = re.search(r'T(\d+)', fname.upper())
            if m:
                task_id = int(m.group(1))
                if 22 <= task_id <= 36:
                    is_fall = True
                elif 1 <= task_id <= 21:
                    is_adl = True
            
            if not (is_fall or is_adl):
                is_fall = 'FALL' in fname.upper() or fname.upper().startswith('F')
                is_adl = 'ADL' in fname.upper() or fname.upper().startswith('A')
            
            if not (is_fall or is_adl):
                parent_dir = os.path.basename(os.path.dirname(fpath)).upper()
                is_fall = 'FALL' in parent_dir or parent_dir.startswith('F')
                is_adl = 'ADL' in parent_dir or parent_dir.startswith('A')
                
            if not (is_fall or is_adl):
                continue
                
            try:
                raw_data = self.parse_file(fpath)
                # Resample 100Hz -> 25Hz
                resampled = resample_signal(raw_data, orig_sr=self.orig_sr, target_sr=self.target_sr)
                
                if is_fall:
                    x_fall, y_imp, _ = extract_fall_windows(
                        resampled,
                        acc_indices=(0, 1, 2),
                        delta=self.delta,
                        theta=self.theta,
                        extract_impact=True
                    )
                    X_list.append(x_fall)
                    y_list.append(1)
                    y_impact_list.append(y_imp)
                    meta_list.append(fname)
                else:
                    adl_wins = extract_adl_windows(
                        resampled,
                        acc_indices=(0, 1, 2),
                        delta=self.delta,
                        theta=self.theta,
                        mode='peak'
                    )
                    for win in adl_wins:
                        X_list.append(win)
                        y_list.append(0)
                        y_impact_list.append(None)
                        meta_list.append(fname)
            except Exception as e:
                continue
                
        return X_list, y_list, y_impact_list, meta_list
