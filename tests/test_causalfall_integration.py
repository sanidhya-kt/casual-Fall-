import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.causalfall import CausalFall
from training.loss import CausalFallLoss

def test_causalfall_integration():
    B = 128
    C = 9
    delta = 72
    
    # 1. Initialize model
    model = CausalFall(
        num_channels=C,
        delta=delta,
        num_heads=9,
        num_mha_layers=3,
        dim_feedforward=16,
        reduced_dim=4,
        intervention_strategy="zero"
    )
    
    # 2. Synthetic batch
    x = torch.randn(B, C, delta)
    y_true = torch.randint(0, 2, (B,))
    
    # 3. Forward pass
    outputs = model(x)
    
    # Verify shapes
    assert outputs['y_o'].shape == (B, 2), f"Expected y_o shape ({B}, 2), got {outputs['y_o'].shape}"
    assert outputs['y_causal'].shape == (B, 2), f"Expected y_causal shape ({B}, 2), got {outputs['y_causal'].shape}"
    assert outputs['y_cf'].shape == (B, 2), f"Expected y_cf shape ({B}, 2), got {outputs['y_cf'].shape}"
    assert outputs['st'].shape == (B, C, delta), f"Expected st shape ({B}, {C}, {delta}), got {outputs['st'].shape}"
    assert outputs['c'].shape == (B, C, delta), f"Expected c shape ({B}, {C}, {delta}), got {outputs['c'].shape}"
    assert outputs['f_co'].shape == (B, C, delta), f"Expected f_co shape ({B}, {C}, {delta}), got {outputs['f_co'].shape}"
    
    # 4. Loss computation
    loss_fn = CausalFallLoss()
    loss_global, loss_dec, loss_count = loss_fn(
        outputs['y_o'],
        outputs['y_causal'],
        outputs['y_cf'],
        y_true
    )
    
    # 5. Backpropagation
    loss_global.backward()
    
    # Verify parameter gradients
    assert model.stae.layers[0].q_proj.weight.grad is not None
    assert model.cee.W_Scale.grad is not None
    assert model.state_decoder.L1.weight.grad is not None
    assert model.state_decoder.L2.weight.grad is not None
    
    print(f"loss_global = {loss_global.item():.4f}, loss_decoder = {loss_dec.item():.4f}, loss_counter = {loss_count.item():.4f}")
    print("[PASS] test_causalfall_integration passed successfully.")

if __name__ == '__main__':
    test_causalfall_integration()
