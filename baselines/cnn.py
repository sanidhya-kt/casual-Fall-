import torch
import torch.nn as nn

class CNNBaseline(nn.Module):
    """
    CNN Baseline for fall prediction based on Zhang & Zhu (2018).
    Applies multi-layer 1D convolutions across channels and time.
    """
    def __init__(self, in_channels: int = 9, delta: int = 72, num_classes: int = 2):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, delta)
        feat = self.conv_net(x).squeeze(-1)  # (B, 128)
        logits = self.fc(feat)               # (B, num_classes)
        return logits
