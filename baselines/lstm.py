import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    """
    LSTM Baseline based on Musci et al. (2020).
    Processes temporal sequence of IMU channels.
    """
    def __init__(self, in_channels: int = 9, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, delta) -> permute to (B, delta, C) for LSTM
        x_seq = x.permute(0, 2, 1)
        lstm_out, (hn, _) = self.lstm(x_seq)
        # Take last time step
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        return logits


class BiLSTMBaseline(nn.Module):
    """
    Bidirectional LSTM Baseline based on Mubibya et al. (2023).
    Captures forward and backward temporal sequences.
    """
    def __init__(self, in_channels: int = 9, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, delta) -> (B, delta, C)
        x_seq = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_seq)
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        return logits
