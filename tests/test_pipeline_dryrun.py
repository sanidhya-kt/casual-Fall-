import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import FallDataset, create_train_test_split
from causal.granger_causality import compute_causal_matrix
from models.causalfall import CausalFall
from training.trainer import Trainer

def test_dryrun_pipeline():
    print("Testing end-to-end training and evaluation dry run...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    N_samples = 40
    C = 9
    delta = 72
    
    # Generate synthetic fall (y=1) and ADL (y=0) windows
    X_list = [np.random.randn(C, delta).astype(np.float32) for _ in range(N_samples)]
    y_list = [1 if i < 20 else 0 for i in range(N_samples)]
    impact_list = [np.random.randn(C, delta).astype(np.float32) if y_list[i] == 1 else None for i in range(N_samples)]
    
    train_ds, test_ds, normalizer, train_impacts, train_falls = create_train_test_split(
        X_list, y_list, impact_list, train_ratio=0.75, random_seed=42
    )
    
    assert len(train_ds) == 30
    assert len(test_ds) == 10
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # Compute Granger causality matrix
    E_mat, channel_weights = compute_causal_matrix(train_falls, train_impacts, max_lag=3, threshold=0.5)
    assert E_mat.shape == (9, 9)
    assert len(channel_weights) == 9
    
    # Initialize CausalFall
    model = CausalFall(
        num_channels=9,
        delta=delta,
        num_heads=9,
        num_mha_layers=3,
        dim_feedforward=16,
        reduced_dim=4,
        channel_weights=channel_weights,
        intervention_strategy="zero"
    )
    
    # Run 3 training epochs
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=1e-3,
        epochs=3,
        checkpoint_dir="./checkpoints/test_dryrun",
        is_causalfall=True
    )
    
    results = trainer.train(verbose=True)
    metrics = results['final_metrics']
    print("Dry run results:", metrics)
    assert 'Accuracy' in metrics and 'F1-Score' in metrics
    print("[PASS] test_dryrun_pipeline passed successfully.")

if __name__ == '__main__':
    test_dryrun_pipeline()
