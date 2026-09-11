import torch
import torch.nn as nn
from typing import Tuple

class CausalFallLoss(nn.Module):
    """
    Loss functions for CausalFall.
    [PAPER SPECIFICATION] Section 4.3.2, Eq. (20)-(23):
        - L_Counter = MSE(Y_o, Y_Causal) = MSE(SD(F_Cf), 0)
        - L_Decoder = CrossEntropy(Y_o, y_true)
        - L_Global = L_Decoder + L_Counter
    """
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        y_o: torch.Tensor,
        y_causal: torch.Tensor,
        y_cf: torch.Tensor,
        y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            y_o: Tensor of shape (B, 2) original prediction probabilities or logits
            y_causal: Tensor of shape (B, 2) debiased prediction
            y_cf: Tensor of shape (B, 2) counterfactual prediction SD(F_Cf)
            y_true: Tensor of shape (B,) ground truth class labels (0 or 1)
            
        Returns:
            loss_global: L_Decoder + L_Counter (Eq. 23)
            loss_decoder: Cross-Entropy loss (Eq. 21-22)
            loss_counter: L2 / MSE loss (Eq. 20)
        """
        # [PAPER SPECIFICATION] Eq. (21)-(22): Cross-Entropy Loss on original prediction
        # For numerical stability with probabilities y_o, clamp probabilities or pass logits
        eps = 1e-7
        y_o_clamped = torch.clamp(y_o, eps, 1.0 - eps)
        # Using negative log likelihood:
        loss_decoder = nn.functional.nll_loss(torch.log(y_o_clamped), y_true)
        
        # [PAPER SPECIFICATION] Eq. (20): L2 loss between Y_o and Y_causal
        # Note: Y_o - Y_causal = y_cf = SD(F_Cf)
        loss_counter = self.mse_loss(y_o, y_causal)
        
        # [PAPER SPECIFICATION] Eq. (23): L_Global = L_Decoder + L_Counter
        loss_global = loss_decoder + loss_counter
        
        return loss_global, loss_decoder, loss_counter
