import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from causal.granger_causality import compute_bivariate_granger_causality, compute_causal_matrix
from models.cee import CEE

def test_granger_causality():
    # Construct synthetic causal pair: y(t) depends on x(t-1)
    np.random.seed(42)
    N = 72
    x = np.random.randn(N)
    y = np.zeros(N)
    for t in range(1, N):
        y[t] = 0.8 * y[t-1] + 0.9 * x[t-1] + np.random.randn() * 0.1
        
    m_gc = compute_bivariate_granger_causality(x, y, max_lag=2)
    assert m_gc > 1.0, f"Expected strong causality M_GC > 1.0, got {m_gc}"
    
    # Construct uncoupled pair
    z = np.random.randn(N)
    w = np.random.randn(N)
    m_gc_null = compute_bivariate_granger_causality(z, w, max_lag=2)
    assert m_gc_null < m_gc, f"Null causality {m_gc_null} should be much smaller than coupled {m_gc}"
    print("[PASS] test_granger_causality passed.")


def test_causal_matrix_and_cee():
    # 5 fall samples, 9 channels, 72 timesteps
    np.random.seed(42)
    falls = np.random.randn(5, 9, 72)
    impacts = np.random.randn(5, 9, 72)
    # Inject strong causal connection from channel 2 in falling to channel 2 in impact
    impacts[:, 2, 1:] += 0.8 * falls[:, 2, :-1]
    
    E_mat, w = compute_causal_matrix(falls, impacts, max_lag=3, threshold=0.5)
    assert E_mat.shape == (9, 9), f"Expected (9, 9), got {E_mat.shape}"
    assert len(w) == 9, f"Expected length 9, got {len(w)}"
    assert np.isclose(np.sum(w), 1.0), f"Weights should sum to 1, got {np.sum(w)}"
    
    # Test CEE module forward and backprop
    cee = CEE(num_channels=9, channel_weights=w)
    x = torch.randn(128, 9, 72)
    out = cee(x)
    assert out.shape == (128, 9, 72), f"Expected (128, 9, 72), got {out.shape}"
    
    loss = out.sum()
    loss.backward()
    assert cee.W_Scale.grad is not None and torch.norm(cee.W_Scale.grad) > 0, "CEE W_Scale gradients missing!"
    print("[PASS] test_causal_matrix_and_cee passed.")


if __name__ == '__main__':
    test_granger_causality()
    test_causal_matrix_and_cee()
    print("\nALL CEE & GRANGER CAUSALITY TESTS PASSED SUCCESSFULLY!")
