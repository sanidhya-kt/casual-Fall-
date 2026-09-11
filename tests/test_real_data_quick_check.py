import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import load_preprocessed_dataset
from causal.granger_causality import compute_causal_matrix
from models.causalfall import CausalFall
from baselines import (
    CNNBaseline,
    LSTMBaseline,
    BiLSTMBaseline,
    ConvLSTMBaseline,
    GCNLSTMBaseline,
    TransformerBaseline
)
from training.trainer import Trainer

def test_real_data_quick_check():
    print("=== Testing Quick Verification on Real Preprocessed Data ===")
    cache_path = "./data/cache/kfall_preprocessed.npz"
    if not os.path.exists(cache_path):
        print(f"[SKIP] Cache not found at {cache_path}")
        return

    # Load small subset: 60 samples
    data = np.load(cache_path, allow_pickle=True)
    X_sub = data['X'][:60]
    y_sub = data['y'][:60]
    impact_sub = data['impact'][:60]
    impact_list = [impact_sub[i] if y_sub[i] == 1 else None for i in range(len(y_sub))]
    
    from data.dataset import create_train_test_split
    train_ds, test_ds, normalizer, train_impacts, train_falls = create_train_test_split(
        X_sub, y_sub, impact_list, train_ratio=0.7, random_seed=42, normalize=True
    )
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # Precompute Granger causality on sample falls
    print("Computing Granger causality on real fall samples...")
    if train_falls is not None and train_impacts is not None and len(train_falls) > 0:
        E_mat, channel_weights = compute_causal_matrix(train_falls[:5], train_impacts[:5], max_lag=3, threshold=0.5)
    else:
        channel_weights = np.ones(9) / 9.0
    print(f"Sample channel weights: {np.round(channel_weights, 4)}")
    
    # 1. Test CausalFall
    print("\nTesting CausalFall for 2 epochs...")
    model = CausalFall(
        num_channels=9,
        delta=72,
        num_heads=9,
        num_mha_layers=3,
        dim_feedforward=16,
        reduced_dim=4,
        channel_weights=channel_weights,
        intervention_strategy="zero"
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=1e-3,
        epochs=2,
        checkpoint_dir="./checkpoints/quick_test_causalfall",
        is_causalfall=True
    )
    res = trainer.train(verbose=True)
    print("CausalFall test passed. Metrics:", res['final_metrics'])
    
    # 2. Test Baselines
    print("\nTesting all 6 Baselines for 1 epoch each...")
    baselines = {
        'CNN': CNNBaseline(in_channels=9, delta=72),
        'LSTM': LSTMBaseline(in_channels=9),
        'Bi-LSTM': BiLSTMBaseline(in_channels=9),
        'ConvLSTM': ConvLSTMBaseline(in_channels=9),
        'GCN-LSTM': GCNLSTMBaseline(num_nodes=9),
        'Transformer': TransformerBaseline(in_channels=9, delta=72)
    }
    for b_name, b_mod in baselines.items():
        b_trainer = Trainer(
            model=b_mod,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=1e-3,
            epochs=1,
            checkpoint_dir=f"./checkpoints/quick_test_{b_name.lower()}",
            is_causalfall=False
        )
        b_res = b_trainer.train(verbose=False)
        print(f"[{b_name}] F1: {b_res['final_metrics']['F1-Score']:.2f}% | Acc: {b_res['final_metrics']['Accuracy']:.2f}%")
        
    print("\n[PASS] All models successfully verified on real preprocessed data!")

if __name__ == '__main__':
    test_real_data_quick_check()
