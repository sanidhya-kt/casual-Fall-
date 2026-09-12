import os
import sys
import argparse
import csv
import json
import random
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.sisfall_loader import SisFallLoader
from data.kfall_loader import KFallLoader
from data.dataset import create_train_val_test_split
from causal.granger_causality import compute_causal_matrix
from models.causalfall import CausalFall
from training.trainer import Trainer

def run_experiment(
    config_path: str,
    data_override_dir: str = None,
    epochs_override: int = None,
    seed_override: int = None,
    split_mode_override: str = None,
    max_lag_override: int = None,
    normalize_override: bool = None,
):
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
    val_ratio = cfg.get('training', {}).get('val_ratio', 0.16)
    random_seed = seed_override if seed_override is not None else cfg.get('training', {}).get('random_seed', 42)
    split_mode = split_mode_override or cfg.get('training', {}).get('split_mode', 'file')
    normalize = normalize_override if normalize_override is not None else cfg.get('training', {}).get('normalize', True)
    max_lag = max_lag_override if max_lag_override is not None else cfg.get('causal', {}).get('max_lag', 5)
    causal_threshold = cfg.get('causal', {}).get('threshold', 1.0)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    print(f"=== Running CausalFall Main Experiment ===")
    print(f"Dataset: {dataset_name} | Root: {root_dir}")
    print(f"Settings: delta={delta}, theta={theta}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    cache_path = f"./data/cache/{dataset_name.lower()}_preprocessed.npz"
    if os.path.exists(cache_path):
        print(f"Loading preprocessed dataset from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        meta = data['meta'].tolist() if 'meta' in data else None
        impact = data['impact']
        labels = data['y']
        impact_list = [impact[i] if labels[i] == 1 else None for i in range(len(labels))]
        train_ds, val_ds, test_ds, normalizer, train_impacts, train_falls, split_metadata = create_train_val_test_split(
            data['X'], labels, impact_list, meta_list=meta,
            train_ratio=train_ratio, val_ratio=val_ratio,
            random_seed=random_seed, normalize=normalize, split_mode=split_mode
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
        train_ds, val_ds, test_ds, normalizer, train_impacts, train_falls, split_metadata = create_train_val_test_split(
            X_list, y_list, impact_list, meta_list=meta_list,
            train_ratio=train_ratio, val_ratio=val_ratio,
            random_seed=random_seed, normalize=normalize, split_mode=split_mode
        )
    print(f"Split mode: {split_mode} | {json.dumps(split_metadata, sort_keys=True)}")
    print(f"Normalization: {'train-fitted channel z-score' if normalize else 'none'}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 3. Precompute Granger causality offline [PAPER SPECIFICATION] Sec 5.4.1
    print("Precomputing Granger Causality matrix across training fall samples...")
    if train_falls is not None and train_impacts is not None and len(train_falls) > 0:
        E_mat, channel_weights = compute_causal_matrix(
            train_falls, train_impacts, max_lag=max_lag, threshold=causal_threshold
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
        val_loader=val_loader,
        learning_rate=lr,
        epochs=epochs,
        checkpoint_dir=f"./checkpoints/{dataset_name}_causalfall",
        is_causalfall=True
    )
    
    results = trainer.train()
    print(f"\n=== Best Validation Metrics (epoch {results['best_epoch']}) ===")
    for k, v in results['best_validation_metrics'].items():
        if k != 'confusion_matrix':
            print(f"{k}: {v:.2f}%")
    print("\n=== Final Test Metrics (test evaluated once) ===")
    for k, v in results['final_test_metrics'].items():
        print(f"{k}: {v if k == 'confusion_matrix' else f'{v:.2f}%'}")

    report_dir = os.path.join('reports', 'runs')
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f'{dataset_name.lower()}_seed_{random_seed}.json'), 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'seed': random_seed,
            'split': split_metadata,
            'normalization': 'train_zscore' if normalize else 'none',
            'max_lag': max_lag,
            'causal_threshold': causal_threshold,
            'best_epoch': results['best_epoch'],
            'validation_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results['best_validation_metrics'].items()},
            'test_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results['final_test_metrics'].items()},
        }, f, indent=2)

    test_cm = results['final_test_metrics']['confusion_matrix']
    val_metrics = results['best_validation_metrics']
    test_metrics = results['final_test_metrics']
    counts = split_metadata
    row = {
        'dataset': dataset_name, 'seed': random_seed, 'split_mode': split_mode,
        'normalization': 'train_zscore' if normalize else 'none', 'max_lag': max_lag,
        'train_total': counts['train']['total'], 'train_falls': counts['train']['falls'], 'train_adls': counts['train']['adls'],
        'val_total': counts['validation']['total'], 'val_falls': counts['validation']['falls'], 'val_adls': counts['validation']['adls'],
        'test_total': counts['test']['total'], 'test_falls': counts['test']['falls'], 'test_adls': counts['test']['adls'],
        'best_val_epoch': results['best_epoch'],
        'val_accuracy': val_metrics['Accuracy'], 'val_precision': val_metrics['Precision'], 'val_recall': val_metrics['Recall'], 'val_f1': val_metrics['F1-Score'],
        'test_accuracy': test_metrics['Accuracy'], 'test_precision': test_metrics['Precision'], 'test_recall': test_metrics['Recall'], 'test_f1': test_metrics['F1-Score'],
        'tp': int(test_cm[1, 1]), 'tn': int(test_cm[0, 0]), 'fp': int(test_cm[0, 1]), 'fn': int(test_cm[1, 0]),
    }
    csv_path = os.path.join('reports', 'experiment_results.csv')
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/kfall.yaml')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--split-mode', choices=['file', 'window', 'subject'], default=None)
    parser.add_argument('--max-lag', type=int, choices=[5, 10, 15], default=None)
    parser.add_argument('--no-normalize', action='store_true')
    args = parser.parse_args()
    
    run_experiment(
        args.config, args.data_dir, args.epochs,
        seed_override=args.seed,
        split_mode_override=args.split_mode,
        max_lag_override=args.max_lag,
        normalize_override=False if args.no_normalize else None,
    )
