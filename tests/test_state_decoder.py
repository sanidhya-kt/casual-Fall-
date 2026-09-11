import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.state_decoder import StateDecoder

def test_state_decoder():
    B = 128
    C = 9
    delta = 72
    reduced_dim = 4
    
    st = torch.randn(B, C, delta)
    c = torch.randn(B, C, delta)
    x = torch.randn(B, C, delta)
    
    sd = StateDecoder(num_channels=C, delta=delta, reduced_dim=reduced_dim)
    y_prob = sd(st, c, x)
    
    assert y_prob.shape == (B, 2), f"Expected shape ({B}, 2), got {y_prob.shape}"
    # Softmax check: probabilities sum to 1
    prob_sums = torch.sum(y_prob, dim=-1)
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5), "Softmax probabilities do not sum to 1!"
    
    # Backprop test
    loss = y_prob.sum()
    loss.backward()
    assert sd.L1.weight.grad is not None, "L1 gradients missing!"
    assert sd.L2.weight.grad is not None, "L2 gradients missing!"
    print("[PASS] test_state_decoder passed.")

if __name__ == '__main__':
    test_state_decoder()
