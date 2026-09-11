import math
import torch
import torch.nn as nn
from typing import Optional

class PositionalEmbedding(nn.Module):
    """
    Positional Embedding as defined in [PAPER SPECIFICATION] Section 4.1.1, Eq. (2):
        PE(x_t) = sin(omega_k * t) if t = 2k
        PE(x_t) = cos(omega_k * t) if t = 2k + 1
        where omega_k = 1 / T^(2k / delta), k in {0, 1, ..., floor(delta / 2)}
    """
    def __init__(self, delta: int = 72, channels: int = 9, pe_base: float = 10000.0):
        super().__init__()
        self.delta = delta
        self.channels = channels
        
        # Precompute PE tensor: shape (channels, delta) or (delta, channels)
        pe = torch.zeros(delta, channels)
        position = torch.arange(0, delta, dtype=torch.float32).unsqueeze(1) # (delta, 1)
        
        # omega_k = 1 / T^(2k / delta)
        # Using dimension along channels:
        div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * -(math.log(pe_base) / delta))
        
        pe[:, 0::2] = torch.sin(position * div_term[:(channels + 1) // 2])
        if channels > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:channels // 2])
            
        # Register as non-trainable buffer with shape (1, channels, delta)
        # to match input X of shape (B, C, delta)
        self.register_buffer('pe', pe.T.unsqueeze(0)) # (1, C, delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, delta)
        Returns:
            x + pe: Tensor of shape (B, C, delta)
        """
        return x + self.pe[:, :, :x.size(2)]


class STAELayer(nn.Module):
    """
    Single layer of Spatio-Temporal Attention Encoder.
    Consists of Multi-Head Self-Attention (scaled by sqrt(delta)),
    residual connection, layer normalization, and feed-forward network with dim_feedforward=16.
    """
    def __init__(
        self,
        embed_dim: int = 72,
        num_heads: int = 9,
        dim_feedforward: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # W_Att (Eq. 4)
        
        # Scaling factor is sqrt(delta) as per [PAPER SPECIFICATION] Eq. (3)
        self.scale = math.sqrt(embed_dim)
        
        # Feed-forward network with dim_feedforward=16 [PAPER SPECIFICATION] Sec 5.2.3
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, delta) where C is sequence length and delta is embed_dim,
               or (B, C, delta) representing multi-channel temporal data.
        Returns:
            Tensor of shape (B, C, delta)
        """
        B, C, D = x.shape
        
        # Multi-head self-attention across channels/time
        residual = x
        q = self.q_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, C, head_dim)
        k = self.k_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, C, head_dim)
        v = self.v_proj(x).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, C, head_dim)
        
        # Scaled dot-product attention [PAPER SPECIFICATION] Eq. (3)
        # S = QK^T / sqrt(delta)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, C, C)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)  # (B, H, C, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, C, D)  # Concatenation of heads (Eq. 4)
        out = self.out_proj(out)  # W_Att
        
        # Residual + Norm
        x = self.norm1(residual + self.dropout(out))
        
        # FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class STAE(nn.Module):
    """
    Spatio-Temporal Attention Encoder (STAE).
    [PAPER SPECIFICATION] Section 4.1 & Section 5.2.3:
        - Positional Embedding (PE)
        - 3 MHA layers (num_mha_layers = 3)
        - H = 9 attention heads
        - dim_feedforward = 16
        - delta = 72
        - Output shape: (B, C, delta)
    """
    def __init__(
        self,
        num_channels: int = 9,
        delta: int = 72,
        num_heads: int = 9,
        num_layers: int = 3,
        dim_feedforward: int = 16,
        dropout: float = 0.1,
        pe_base: float = 10000.0
    ):
        super().__init__()
        self.num_channels = num_channels
        self.delta = delta
        
        # Positional embedding
        self.pe = PositionalEmbedding(delta=delta, channels=num_channels, pe_base=pe_base)
        
        # 3 MHA layers with dim_feedforward=16
        self.layers = nn.ModuleList([
            STAELayer(
                embed_dim=delta,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, delta)
        Returns:
            ST: Tensor of shape (B, C, delta) representing spatio-temporal features
        """
        # Inject positional embedding: x_pe = x + PE(x)
        x = self.pe(x)
        
        # Pass through 3 MHA layers
        for layer in self.layers:
            x = layer(x)
            
        return x
