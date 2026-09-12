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
from data.dataset import create_train_val_test_split
from baselines import (
    CNNBaseline,
    LSTMBaseline,
    BiLSTMBaseline,
    ConvLSTMBaseline,
    GCNLSTMBaseline,
    TransformerBaseline
)
from training.trainer import Trainer

def run_all_baselines(config_path: str, data_override_dir: str = None, epochs_override: int = None):
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
    random_seed = cfg.get('training', {}).get('random_seed', 42)
    split_mode = cfg.get('training', {}).get('split_mode', 'file')
    normalize = cfg.get('training', {}).get('normalize', True)
    
    cache_path = f"./data/cache/{dataset_name.lower()}_preprocessed.npz"
    if os.path.exists(cache_path):
        print(f"Loading preprocessed dataset from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        labels = data['y']
        impact = data['impact']
        impact_list = [impact[i] if labels[i] == 1 else None for i in range(len(labels))]
        train_ds, val_ds, test_ds, normalizer, _, _, _ = create_train_val_test_split(
            data['X'], labels, impact_list,
            meta_list=data['meta'].tolist() if 'meta' in data else None,
            train_ratio=train_ratio, val_ratio=val_ratio,
            random_seed=random_seed, normalize=normalize, split_mode=split_mode
        )
    else:
        # 1. Load dataset
        if dataset_name.lower() == 'sisfall':
            loader = SisFallLoader(root_dir=root_dir, delta=delta, theta=theta)
        else:
            loader = KFallLoader(root_dir=root_dir, delta=delta, theta=theta)
            
        if not os.path.exists(root_dir) or len(os.listdir(root_dir)) == 0:
            print(f"[WARNING] Dataset path '{root_dir}' is empty or does not exist.")
            return None
            
        X_list, y_list, impact_list, meta_list = loader.load_dataset()
        train_ds, val_ds, test_ds, normalizer, _, _, _ = create_train_val_test_split(
            X_list, y_list, impact_list, meta_list=meta_list,
            train_ratio=train_ratio, val_ratio=val_ratio,
            random_seed=random_seed, normalize=normalize, split_mode=split_mode
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    baselines = {
        'CNN': CNNBaseline(in_channels=9, delta=delta),
        'LSTM': LSTMBaseline(in_channels=9),
        'Bi-LSTM': BiLSTMBaseline(in_channels=9),
        'ConvLSTM': ConvLSTMBaseline(in_channels=9),
        'GCN-LSTM': GCNLSTMBaseline(num_nodes=9),
        'Transformer': TransformerBaseline(in_channels=9, delta=delta)
    }
    
    results = {}
    print(f"\n{'='*50}\nEvaluating All Baseline Models on {dataset_name}\n{'='*50}")
    
    for name, model in baselines.items():
        print(f"\n--- Training {name} ---")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            learning_rate=lr,
            epochs=epochs,
            checkpoint_dir=f"./checkpoints/{dataset_name}_{name.lower()}",
            is_causalfall=False
        )
        res = trainer.train()
        results[name] = res['best_metrics']
        
    print("\n" + "="*70)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<15} | {m['Accuracy']:<10.2f} | {m['Precision']:<10.2f} | {m['Recall']:<10.2f} | {m['F1-Score']:<10.2f}")
    print("=" * 70)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/kfall.yaml')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    run_all_baselines(args.config, args.data_dir, args.epochs)
