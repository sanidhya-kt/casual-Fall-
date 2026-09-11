import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.loss import CausalFallLoss

def test_causalfall_loss():
    B = 128
    y_o = torch.softmax(torch.randn(B, 2), dim=-1)
    y_cf = torch.softmax(torch.randn(B, 2), dim=-1)
    y_causal = y_o - y_cf
    y_true = torch.randint(0, 2, (B,))
    
    loss_fn = CausalFallLoss()
    loss_global, loss_dec, loss_count = loss_fn(y_o, y_causal, y_cf, y_true)
    
    assert loss_global > 0, "Global loss must be positive"
    assert loss_dec > 0, "Decoder loss must be positive"
    assert loss_count >= 0, "Counter loss must be non-negative"
    assert torch.isclose(loss_global, loss_dec + loss_count), "L_Global must equal L_Decoder + L_Counter"
    
    print("[PASS] test_causalfall_loss passed.")

if __name__ == '__main__':
    test_causalfall_loss()
