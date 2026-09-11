import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class CEE(nn.Module):
    """
    Causal Effect Encoder (CEE).
    [PAPER SPECIFICATION] Section 4.2, Eq. (11):
        C : CEE(X^i) = [ sum_j E(L_falling^i, L_impact^j) / sum_{j, k} E(L_falling^j, L_impact^k) ] * W_Scale * X^i
        
    Takes input X in R^(B x C x delta) and weights each channel i by its precomputed
    Granger causal effect relative to all other channels, modulated by learnable parameter W_Scale.
    """
    def __init__(
        self,
        num_channels: int = 9,
        channel_weights: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.num_channels = num_channels
        
        if channel_weights is None:
            # Default to uniform weights 1/C if not precomputed yet
            weights = torch.ones(num_channels, dtype=torch.float32) / num_channels
        else:
            weights = torch.tensor(channel_weights, dtype=torch.float32)
            
        # Register channel causal weights as non-trainable buffer: shape (1, C, 1)
        self.register_buffer('channel_causal_weights', weights.view(1, num_channels, 1))
        
        # W_Scale: learnable scaling parameter as per [PAPER SPECIFICATION] Eq. (11)
        self.W_Scale = nn.Parameter(torch.ones(1, num_channels, 1, dtype=torch.float32))

    def update_weights(self, channel_weights: np.ndarray):
        """Updates precomputed Granger causality weights."""
        weights = torch.tensor(channel_weights, dtype=torch.float32).view(1, self.num_channels, 1)
        self.channel_causal_weights.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, delta)
        Returns:
            C: Tensor of shape (B, C, delta)
        """
        # Eq. (11): weights * W_Scale * X^i
        c_feat = self.channel_causal_weights * self.W_Scale * x
        return c_feat
