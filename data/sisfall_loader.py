import os
import glob
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from preprocessing.resampler import resample_signal
from preprocessing.windowing import extract_fall_windows, extract_adl_windows

class SisFallLoader:
    """
    Dataset loader for SisFall (Sucerquia et al., 2017).
    [PAPER SPECIFICATION] Section 5.1:
        200Hz original sampling frequency.
        38 volunteers (23 adults, 15 elderly).
        19 ADLs, 15 falls, total 4,505 samples.
        Sensors: 2 accelerometers and 1 gyroscope at waist.
        Resampled to 25Hz.
    """
    def __init__(
        self,
        root_dir: str,
        target_sr: int = 25,
        orig_sr: int = 200,
        delta: int = 72,
        theta: int = 15,
        use_channels: int = 9  # 9 channels: ADXL345 (3), ITG3200 (3), MMA8451Q (3)
    ):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.orig_sr = orig_sr
        self.delta = delta
        self.theta = theta
        self.use_channels = use_channels
        
    def parse_file(self, filepath: str) -> np.ndarray:
        """
        Parses a SisFall text file.
        Format has lines with semicolon or comma-separated integers/floats.
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        data_rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove trailing semicolons or commas
            line = line.rstrip(';,')
            # Split by comma or whitespace or semicolon
            if ';' in line:
                tokens = line.split(';')
            elif ',' in line:
                tokens = line.split(',')
            else:
                tokens = line.split()
            try:
                row = [float(t.strip()) for t in tokens if t.strip()]
                if len(row) >= 9:
                    data_rows.append(row[:9])
            except ValueError:
                continue
                
        if not data_rows:
            raise ValueError(f"Could not parse valid data rows from {filepath}")
            
        data = np.array(data_rows, dtype=np.float32)
        if self.use_channels == 6:
            # First accelerometer (0,1,2) and gyroscope (3,4,5)
            data = data[:, :6]
        return data

    def load_dataset(self) -> Tuple[List[np.ndarray], List[int], List[Optional[np.ndarray]], List[str]]:
        """
        Scans root_dir and loads all available SisFall samples.
        
        Returns:
            X_list: list of (C, delta) arrays
            y_list: list of int labels (0 = ADL, 1 = Fall)
            y_impact_list: list of (C, delta) impact arrays for falls (or None for ADLs)
            meta_list: list of file metadata/paths
        """
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"SisFall directory not found at {self.root_dir}")
            
        # Find all .txt files
        file_patterns = [
            os.path.join(self.root_dir, '**', '*.txt'),
            os.path.join(self.root_dir, '*.txt')
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
            # Check if file is ADL (starts with 'D' or contains 'ADL') or Fall (starts with 'F' or contains 'FALL')
            is_fall = fname.startswith('F') or 'FALL' in fname.upper()
            is_adl = fname.startswith('D') or 'ADL' in fname.upper()
            
            if not (is_fall or is_adl):
                continue
                
            try:
                raw_data = self.parse_file(fpath)
                # Resample 200Hz -> 25Hz
                resampled = resample_signal(raw_data, orig_sr=self.orig_sr, target_sr=self.target_sr)
                
                if is_fall:
                    # Accelerometer axes: columns 0, 1, 2
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
                    # ADL sample
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
                # Skip corrupted or unreadable files with notice
                continue
                
        return X_list, y_list, y_impact_list, meta_list
