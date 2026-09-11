import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baselines import (
    CNNBaseline,
    LSTMBaseline,
    BiLSTMBaseline,
    ConvLSTMBaseline,
    GCNLSTMBaseline,
    TransformerBaseline
)

def test_baselines():
    B = 32
    C = 9
    delta = 72
    x = torch.randn(B, C, delta)
    
    models = {
        'CNN': CNNBaseline(in_channels=C, delta=delta),
        'LSTM': LSTMBaseline(in_channels=C),
        'Bi-LSTM': BiLSTMBaseline(in_channels=C),
        'ConvLSTM': ConvLSTMBaseline(in_channels=C),
        'GCN-LSTM': GCNLSTMBaseline(num_nodes=C),
        'Transformer': TransformerBaseline(in_channels=C, delta=delta)
    }
    
    for name, m in models.items():
        out = m(x)
        assert out.shape == (B, 2), f"{name} output shape mismatch: {out.shape}"
        loss = out.sum()
        loss.backward()
        print(f"[PASS] {name} baseline forward & backward verified.")

if __name__ == '__main__':
    test_baselines()
    print("\nALL BASELINE TESTS PASSED SUCCESSFULLY!")
