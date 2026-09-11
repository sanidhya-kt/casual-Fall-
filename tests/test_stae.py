import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.stae import STAE

def test_stae_forward():
    B = 128
    C = 9
    delta = 72
    
    x = torch.randn(B, C, delta)
    model = STAE(num_channels=C, delta=delta, num_heads=9, num_layers=3, dim_feedforward=16)
    out = model(x)
    
    assert out.shape == (B, C, delta), f"Expected shape ({B}, {C}, {delta}), got {out.shape}"
    loss = out.sum()
    loss.backward()
    
    # Check that gradients flowed
    has_grads = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grads, "No gradients computed for STAE parameters!"
    print("[PASS] test_stae_forward passed successfully.")

if __name__ == '__main__':
    test_stae_forward()
