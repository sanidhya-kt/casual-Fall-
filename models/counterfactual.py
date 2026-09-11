import torch
import torch.nn as nn
from typing import Literal

MaskStrategy = Literal["zero", "mean", "random", "none"]

class CounterfactualIntervention(nn.Module):
    """
    Counterfactual Intervention module as defined in [PAPER SPECIFICATION] Section 4.3.2, Eq. (17):
        [X = X']_do = X * M
        M_t = 0 if t in [t_impact - theta - delta, t_impact - theta - delta/2] (first half of falling window)
        M_t = 1 otherwise (second half of falling window)
        
    Also implements the comparison variants from [PAPER SPECIFICATION] Table 7:
        1) No-Intervention
        2) Random-Mask
        3) Mean-Mask
        4) Zero-Mask (proposed)
    """
    def __init__(self, delta: int = 72, strategy: MaskStrategy = "zero"):
        super().__init__()
        self.delta = delta
        self.half_delta = delta // 2  # delta/2 = 36 for delta=72
        self.strategy = strategy
        
        # Binary mask vector M: shape (1, 1, delta)
        mask = torch.ones(1, 1, delta, dtype=torch.float32)
        mask[:, :, :self.half_delta] = 0.0
        self.register_buffer('zero_mask', mask)

    def set_strategy(self, strategy: MaskStrategy):
        self.strategy = strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies intervention [X = X']_do to input tensor X of shape (B, C, delta).
        
        Args:
            x: Tensor of shape (B, C, delta)
        Returns:
            x_intervened: Tensor of shape (B, C, delta)
        """
        if self.strategy == "none" or self.strategy == "No-Intervention":
            return x
            
        elif self.strategy == "zero" or self.strategy == "Zero-Mask":
            # [PAPER SPECIFICATION] Eq. (17): Zero-valued mask on first half
            return x * self.zero_mask[:, :, :x.size(2)]
            
        elif self.strategy == "mean" or self.strategy == "Mean-Mask":
            # [PAPER SPECIFICATION] Table 7: Replace first half with mean values
            # Compute temporal mean per channel: shape (B, C, 1)
            mean_vals = torch.mean(x, dim=2, keepdim=True)
            x_intervened = x.clone()
            x_intervened[:, :, :self.half_delta] = mean_vals.expand(-1, -1, self.half_delta)
            return x_intervened
            
        elif self.strategy == "random" or self.strategy == "Random-Mask":
            # [PAPER SPECIFICATION] Table 7: Replace first half with random values
            x_intervened = x.clone()
            noise = torch.randn(x.size(0), x.size(1), self.half_delta, device=x.device, dtype=x.dtype)
            x_intervened[:, :, :self.half_delta] = noise
            return x_intervened
            
        else:
            raise ValueError(f"Unknown intervention strategy: {self.strategy}")
