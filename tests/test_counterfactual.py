import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.counterfactual import CounterfactualIntervention

def test_counterfactual_intervention():
    B = 128
    C = 9
    delta = 72
    
    x = torch.ones(B, C, delta) * 5.0
    
    # 1. Zero-mask
    ci_zero = CounterfactualIntervention(delta=delta, strategy="zero")
    x_zero = ci_zero(x)
    assert x_zero.shape == (B, C, delta)
    assert torch.all(x_zero[:, :, :36] == 0.0), "Zero-mask did not zero first 36 timesteps!"
    assert torch.all(x_zero[:, :, 36:] == 5.0), "Zero-mask modified second 36 timesteps!"
    
    # 2. Mean-mask
    x_varied = torch.arange(delta, dtype=torch.float32).view(1, 1, delta).expand(B, C, delta)
    mean_val = x_varied.mean(dim=2, keepdim=True)
    ci_mean = CounterfactualIntervention(delta=delta, strategy="mean")
    x_mean = ci_mean(x_varied)
    assert torch.allclose(x_mean[:, :, :36], mean_val[:, :, :36].expand(-1, -1, 36)), "Mean-mask failed!"
    assert torch.allclose(x_mean[:, :, 36:], x_varied[:, :, 36:]), "Mean-mask altered unmasked part!"
    
    # 3. None
    ci_none = CounterfactualIntervention(delta=delta, strategy="none")
    x_none = ci_none(x)
    assert torch.allclose(x_none, x), "No-intervention modified input!"
    
    # 4. Random
    ci_rand = CounterfactualIntervention(delta=delta, strategy="random")
    x_rand = ci_rand(x)
    assert not torch.allclose(x_rand[:, :, :36], x[:, :, :36]), "Random-mask did not randomize!"
    assert torch.allclose(x_rand[:, :, 36:], x[:, :, 36:]), "Random-mask altered unmasked part!"
    
    print("[PASS] test_counterfactual_intervention passed.")

if __name__ == '__main__':
    test_counterfactual_intervention()
