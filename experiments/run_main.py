import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.sisfall_loader import SisFallLoader
from data.kfall_loader import KFallLoader
from data.dataset import create_train_test_split
from causal.granger_causality import compute_causal_matrix
from models.causalfall import CausalFall
from training.trainer import Trainer

def run_experiment(config_path: str, data_override_dir: str = None, epochs_override: int = None):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    dataset_name = cfg.get('dataset', {}).get('name', 'KFall')
    root_dir = data_override_dir or cfg.get('dataset', {}).get('root_dir', f"./data/{dataset_name}")
    
    delta = cfg.get('model', {}).get('window_size', 72)
    theta = cfg.get('model', {}).get('prediction_horizon', 15)
    epochs = epochs_override or cfg.get('training', {}).get('epochs', 500)
    batch_size = cfg.get('training', {}).get('batch_size', 128)
    lr = cfg.get('training', {}).get('learning_rate', 1e-3)
    train_ratio = cfg.get('training', {}).get('train_ratio', 0.8)
    random_seed = cfg.get('training', {}).get('random_seed', 42)
    
    print(f"=== Running CausalFall Main Experiment ===")
    print(f"Dataset: {dataset_name} | Root: {root_dir}")
    print(f"Settings: delta={delta}, theta={theta}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    cache_path = f"./data/cache/{dataset_name.lower()}_preprocessed.npz"
    if os.path.exists(cache_path):
        print(f"Loading preprocessed dataset from cache: {cache_path}")
        from data.dataset import load_preprocessed_dataset
        train_ds, test_ds, normalizer, train_impacts, train_falls = load_preprocessed_dataset(
            cache_path, train_ratio=train_ratio, random_seed=random_seed, normalize=True
        )
    else:
        # 1. Load dataset from raw
        if dataset_name.lower() == 'sisfall':
            loader = SisFallLoader(root_dir=root_dir, delta=delta, theta=theta)
        else:
            loader = KFallLoader(root_dir=root_dir, delta=delta, theta=theta)
            
        if not os.path.exists(root_dir) or len(os.listdir(root_dir)) == 0:
            print(f"[WARNING] Dataset path '{root_dir}' is empty or does not exist.")
            print("Please provide the dataset directory via --data_dir.")
            return None
            
        X_list, y_list, impact_list, meta_list = loader.load_dataset()
        print(f"Loaded {len(X_list)} samples (Falls: {sum(y_list)}, ADLs: {len(y_list) - sum(y_list)})")
        
        # 2. Split dataset
        train_ds, test_ds, normalizer, train_impacts, train_falls = create_train_test_split(
            X_list, y_list, impact_list, train_ratio=train_ratio, random_seed=random_seed
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 3. Precompute Granger causality offline [PAPER SPECIFICATION] Sec 5.4.1
    print("Precomputing Granger Causality matrix across training fall samples...")
    if train_falls is not None and train_impacts is not None and len(train_falls) > 0:
        E_mat, channel_weights = compute_causal_matrix(
            train_falls, train_impacts, max_lag=5, threshold=1.0
        )
    else:
        channel_weights = np.ones(9) / 9.0
    print(f"Channel weights: {np.round(channel_weights, 4)}")
    
    # 4. Initialize CausalFall
    model = CausalFall(
        num_channels=cfg.get('model', {}).get('num_channels', 9),
        delta=delta,
        num_heads=cfg.get('model', {}).get('num_heads', 9),
        num_mha_layers=cfg.get('model', {}).get('num_mha_layers', 3),
        dim_feedforward=cfg.get('model', {}).get('dim_feedforward', 16),
        reduced_dim=cfg.get('model', {}).get('reduced_dim', 4),
        channel_weights=channel_weights,
        intervention_strategy="zero"
    )
    
    # 5. Train and evaluate
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=lr,
        epochs=epochs,
        checkpoint_dir=f"./checkpoints/{dataset_name}_causalfall",
        is_causalfall=True
    )
    
    results = trainer.train()
    print("\n=== Best Evaluation Metrics ===")
    for k, v in results['best_metrics'].items():
        print(f"{k}: {v:.2f}%")
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/kfall.yaml')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    run_experiment(args.config, args.data_dir, args.epochs)
