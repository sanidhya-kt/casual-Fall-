import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.kfall_loader import KFallLoader
from data.sisfall_loader import SisFallLoader

def build_kfall_cache(
    root_dir: str = "./datasets/raw/KFall Dataset/sensor_data",
    output_path: str = "./data/cache/kfall_preprocessed.npz"
):
    print(f"=== Building KFall Preprocessed Cache ===")
    print(f"Source: {root_dir}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    loader = KFallLoader(root_dir=root_dir, target_sr=25, orig_sr=100, delta=72, theta=15)
    X_list, y_list, impact_list, meta_list = loader.load_dataset()
    
    print(f"Total samples extracted: {len(X_list)} (Falls: {sum(y_list)}, ADLs: {len(y_list) - sum(y_list)})")
    
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    
    # Store impacts as array with dummy zeros for ADLs
    impact_arr = np.zeros_like(X_arr)
    for i, imp in enumerate(impact_list):
        if imp is not None:
            impact_arr[i] = imp
            
    np.savez_compressed(
        output_path,
        X=X_arr,
        y=y_arr,
        impact=impact_arr,
        meta=np.array(meta_list)
    )
    print(f"Saved KFall cache to: {output_path} (Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB)")


def build_sisfall_cache(
    root_dir: str = "./datasets/raw/SisFall",
    output_path: str = "./data/cache/sisfall_preprocessed.npz"
):
    print(f"\n=== Building SisFall Preprocessed Cache ===")
    print(f"Source: {root_dir}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    loader = SisFallLoader(root_dir=root_dir, target_sr=25, orig_sr=200, delta=72, theta=15)
    X_list, y_list, impact_list, meta_list = loader.load_dataset()
    
    print(f"Total samples extracted: {len(X_list)} (Falls: {sum(y_list)}, ADLs: {len(y_list) - sum(y_list)})")
    
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    
    impact_arr = np.zeros_like(X_arr)
    for i, imp in enumerate(impact_list):
        if imp is not None:
            impact_arr[i] = imp
            
    np.savez_compressed(
        output_path,
        X=X_arr,
        y=y_arr,
        impact=impact_arr,
        meta=np.array(meta_list)
    )
    print(f"Saved SisFall cache to: {output_path} (Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['kfall', 'sisfall', 'all'], default='all')
    args = parser.parse_args()
    
    if args.dataset in ['kfall', 'all']:
        build_kfall_cache()
    if args.dataset in ['sisfall', 'all']:
        build_sisfall_cache()
