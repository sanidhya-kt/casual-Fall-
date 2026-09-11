import torch
import torch.nn as nn

class ConvLSTMBaseline(nn.Module):
    """
    ConvLSTM Baseline based on Yu et al. (2020).
    Extracts spatial features via Conv1D, followed by LSTM for temporal dependencies.
    """
    def __init__(self, in_channels: int = 9, conv_channels: int = 32, lstm_hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, delta)
        conv_out = self.conv(x)            # (B, conv_channels, delta)
        lstm_in = conv_out.permute(0, 2, 1) # (B, delta, conv_channels)
        lstm_out, _ = self.lstm(lstm_in)
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        return logits


class GCNLSTMBaseline(nn.Module):
    """
    GCN-LSTM Baseline based on Wu et al. (2021).
    Graph Convolution over sensor channels followed by LSTM over time.
    """
    def __init__(self, num_nodes: int = 9, lstm_hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        # Learnable channel-to-channel graph adjacency
        self.adj = nn.Parameter(torch.eye(num_nodes) + 0.1 * torch.ones(num_nodes, num_nodes))
        self.gcn_linear = nn.Linear(1, 16)
        
        self.lstm = nn.LSTM(
            input_size=num_nodes * 16,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, delta)
        B, C, T = x.shape
        # Normalize adjacency matrix
        deg = torch.sum(self.adj, dim=1, keepdim=True)
        adj_norm = self.adj / torch.clamp(deg, min=1e-5)  # (C, C)
        
        # Spatial graph conv at each timestep
        # x: (B, T, C, 1)
        x_reshaped = x.permute(0, 2, 1).unsqueeze(-1)  # (B, T, C, 1)
        # GCN: A * X * W
        gcn_feat = torch.matmul(adj_norm, x_reshaped)  # (B, T, C, 1)
        gcn_feat = torch.relu(self.gcn_linear(gcn_feat))  # (B, T, C, 16)
        
        # Flatten spatial dimension for LSTM
        lstm_in = gcn_feat.view(B, T, C * 16)  # (B, T, C*16)
        lstm_out, _ = self.lstm(lstm_in)
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        return logits


class TransformerBaseline(nn.Module):
    """
    Transformer Baseline based on Liu et al. (2023b) / MCTN.
    Standard Multi-Head Attention without causal decomposition or counterfactual intervention.
    """
    def __init__(
        self,
        in_channels: int = 9,
        delta: int = 72,
        num_heads: int = 9,
        num_layers: int = 3,
        dim_feedforward: int = 16,
        num_classes: int = 2
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=delta,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(in_channels * delta, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, delta)
        tx_out = self.transformer(x)  # (B, C, delta)
        flat = tx_out.reshape(x.size(0), -1)
        logits = self.fc(flat)
        return logits
