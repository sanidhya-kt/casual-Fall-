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

def run_ablations(config_path: str, data_override_dir: str = None, epochs_override: int = None):
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
    
    cache_path = f"./data/cache/{dataset_name.lower()}_preprocessed.npz"
    if os.path.exists(cache_path):
        print(f"Loading preprocessed dataset from cache: {cache_path}")
        from data.dataset import load_preprocessed_dataset
        train_ds, test_ds, normalizer, train_impacts, train_falls = load_preprocessed_dataset(
            cache_path, train_ratio=train_ratio, random_seed=random_seed, normalize=True
        )
    else:
        if dataset_name.lower() == 'sisfall':
            loader = SisFallLoader(root_dir=root_dir, delta=delta, theta=theta)
        else:
            loader = KFallLoader(root_dir=root_dir, delta=delta, theta=theta)
            
        if not os.path.exists(root_dir) or len(os.listdir(root_dir)) == 0:
            print(f"[WARNING] Dataset path '{root_dir}' is empty or does not exist.")
            return None
            
        X_list, y_list, impact_list, meta_list = loader.load_dataset()
        train_ds, test_ds, normalizer, train_impacts, train_falls = create_train_test_split(
            X_list, y_list, impact_list, train_ratio=train_ratio, random_seed=random_seed
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    if train_falls is not None and train_impacts is not None and len(train_falls) > 0:
        _, channel_weights = compute_causal_matrix(train_falls, train_impacts, max_lag=5, threshold=1.0)
    else:
        channel_weights = np.ones(9) / 9.0
        
    # Ablation 1: Intervention Strategies (Table 7 in Paper)
    # [No-Intervention, Random-Mask, Mean-Mask, Zero-Mask]
    mask_strategies = ['none', 'random', 'mean', 'zero']
    table7_results = {}
    
    print("\n" + "="*70)
    print("Running Ablation: Intervention Methods (Paper Table 7)")
    print("="*70)
    for strat in mask_strategies:
        print(f"\n--- Strategy: {strat.upper()} ---")
        model = CausalFall(
            num_channels=9,
            delta=delta,
            num_heads=9,
            num_mha_layers=3,
            dim_feedforward=16,
            reduced_dim=4,
            channel_weights=channel_weights,
            intervention_strategy=strat,
            use_cee=True,
            use_counterfactual=(strat != 'none')
        )
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=lr,
            epochs=epochs,
            checkpoint_dir=f"./checkpoints/{dataset_name}_ablation_mask_{strat}",
            is_causalfall=True
        )
        res = trainer.train()
        table7_results[strat] = res['best_metrics']
        
    # Ablation 2: With vs Without CEE (Table 5 in Paper)
    print("\n" + "="*70)
    print("Running Ablation: With vs Without CEE (Paper Table 5)")
    print("="*70)
    table5_results = {}
    for use_cee in [True, False]:
        tag = "with_CEE" if use_cee else "without_CEE"
        print(f"\n--- Model: {tag} ---")
        model = CausalFall(
            num_channels=9,
            delta=delta,
            num_heads=9,
            num_mha_layers=3,
            dim_feedforward=16,
            reduced_dim=4,
            channel_weights=channel_weights,
            intervention_strategy="zero",
            use_cee=use_cee,
            use_counterfactual=True
        )
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=lr,
            epochs=epochs,
            checkpoint_dir=f"./checkpoints/{dataset_name}_ablation_{tag}",
            is_causalfall=True
        )
        res = trainer.train()
        table5_results[tag] = res['best_metrics']
        
    # Print Table 7 formatted
    print("\n" + "="*70)
    print("TABLE 7 REPLICATION RESULTS: Intervention Methods")
    print("-" * 70)
    print(f"{'Method':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for strat, m in table7_results.items():
        name_map = {'none': 'No-Intervention', 'random': 'Random-Mask', 'mean': 'Mean-Mask', 'zero': 'Zero-Mask'}
        print(f"{name_map[strat]:<20} | {m['Accuracy']:<10.2f} | {m['Precision']:<10.2f} | {m['Recall']:<10.2f} | {m['F1-Score']:<10.2f}")
        
    # Print Table 5 formatted
    print("\n" + "="*70)
    print("TABLE 5 REPLICATION RESULTS: CEE Impact")
    print("-" * 70)
    print(f"{'Condition':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for tag, m in table5_results.items():
        print(f"{tag:<20} | {m['Accuracy']:<10.2f} | {m['Precision']:<10.2f} | {m['Recall']:<10.2f} | {m['F1-Score']:<10.2f}")
    print("=" * 70)
    
    return table7_results, table5_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/kfall.yaml')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    run_ablations(args.config, args.data_dir, args.epochs)
