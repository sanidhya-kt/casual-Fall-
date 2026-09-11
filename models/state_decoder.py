import torch
import torch.nn as nn

class StateDecoder(nn.Module):
    """
    State Decoder (SD).
    [PAPER SPECIFICATION] Section 4.3.1, Eq. (12)-(16):
        - F_Co = ST + C + X  [Shape: (B, C, delta)]
        - F_L1 = W_L1 * F_Co + B_L1  [Dimension reduction C -> Gamma, Gamma < C]
        - F_GAP = 1 / (Gamma * delta) * sum_{i,j} F_L1  [Shape: (B, 1)]
        - F_L2 = W_L2 * F_GAP + B_L2  [Shape: (B, 2)]
        - y = Softmax(F_L2)  [Shape: (B, 2)]
    """
    def __init__(
        self,
        num_channels: int = 9,
        delta: int = 72,
        reduced_dim: int = 4
    ):
        super().__init__()
        self.num_channels = num_channels
        self.delta = delta
        self.reduced_dim = reduced_dim
        assert reduced_dim < num_channels, f"reduced_dim ({reduced_dim}) must be < num_channels ({num_channels})"
        
        # Linear layer L1: reduces channel dimension C -> Gamma
        # Operating on channel axis: shape (B, C, delta) -> (B, Gamma, delta)
        self.L1 = nn.Conv1d(in_channels=num_channels, out_channels=reduced_dim, kernel_size=1)
        
        # Linear layer L2: maps F_GAP (scalar per sample) to 2 classes (fall vs non-fall)
        self.L2 = nn.Linear(in_features=1, out_features=2)
        
    def forward_features(self, f_co: torch.Tensor) -> torch.Tensor:
        """
        Passes combined features through L1, GAP, and L2 to get logits.
        Args:
            f_co: Tensor of shape (B, C, delta)
        Returns:
            f_l2: Tensor of shape (B, 2)
        """
        # Eq. (13): F_L1 = W_L1 * F_Co + B_L1
        f_l1 = self.L1(f_co)  # (B, Gamma, delta)
        
        # Eq. (14): F_GAP = 1 / (Gamma * delta) * sum_{i,j} F_L1
        # Global Average Pooling across both channel (dim 1) and time (dim 2)
        f_gap = torch.mean(f_l1, dim=(1, 2), keepdim=True)  # (B, 1, 1)
        f_gap = f_gap.squeeze(-1)  # (B, 1)
        
        # Eq. (15): F_L2 = W_L2 * F_GAP + B_L2
        f_l2 = self.L2(f_gap)  # (B, 2)
        return f_l2

    def forward(
        self,
        st: torch.Tensor,
        c: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            st: Tensor of shape (B, C, delta) from STAE
            c: Tensor of shape (B, C, delta) from CEE
            x: Tensor of shape (B, C, delta) original input
        Returns:
            y: Softmax probabilities of shape (B, 2)
        """
        # Eq. (12): F_Co = ST + C + X
        f_co = st + c + x  # (B, C, delta)
        
        # Pass through L1, GAP, L2
        f_l2 = self.forward_features(f_co)
        
        # Eq. (16): Softmax
        y = torch.softmax(f_l2, dim=-1)
        return y
