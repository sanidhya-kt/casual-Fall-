import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any

from models.stae import STAE
from models.cee import CEE
from models.state_decoder import StateDecoder
from models.counterfactual import CounterfactualIntervention, MaskStrategy

class CausalFall(nn.Module):
    """
    CausalFall: Complete end-to-end architecture as defined in Liao et al. (ESWA 2026).
    
    Architecture consists of:
    1. Spatio-Temporal Attention Encoder (STAE): 3 MHA layers, H=9 heads, dim_feedforward=16.
    2. Causal Effect Encoder (CEE): Precomputed Granger causality channel weights + W_Scale.
    3. State Decoder (SD): Combine-Reduction-Decoding, GAP, and Softmax classification.
    4. Counterfactual Intervention: Zero-masking initial falling phase [0, delta/2].
    5. Debiasing: Y_Causal = Y_o - SD(F_Cf).
    """
    def __init__(
        self,
        num_channels: int = 9,
        delta: int = 72,
        num_heads: int = 9,
        num_mha_layers: int = 3,
        dim_feedforward: int = 16,
        reduced_dim: int = 4,
        dropout: float = 0.1,
        pe_base: float = 10000.0,
        channel_weights: Optional[np.ndarray] = None,
        intervention_strategy: MaskStrategy = "zero",
        use_cee: bool = True,
        use_counterfactual: bool = True
    ):
        super().__init__()
        self.num_channels = num_channels
        self.delta = delta
        self.use_cee = use_cee
        self.use_counterfactual = use_counterfactual
        
        # 1. STAE Backbone
        self.stae = STAE(
            num_channels=num_channels,
            delta=delta,
            num_heads=num_heads,
            num_layers=num_mha_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pe_base=pe_base
        )
        
        # 2. CEE Module
        self.cee = CEE(
            num_channels=num_channels,
            channel_weights=channel_weights
        )
        
        # 3. State Decoder
        self.state_decoder = StateDecoder(
            num_channels=num_channels,
            delta=delta,
            reduced_dim=reduced_dim
        )
        
        # 4. Counterfactual Intervention
        self.counterfactual = CounterfactualIntervention(
            delta=delta,
            strategy=intervention_strategy
        )

    def set_cee_weights(self, weights: np.ndarray):
        """Updates precomputed Granger causality channel weights."""
        self.cee.update_weights(weights)

    def set_intervention_strategy(self, strategy: MaskStrategy):
        """Changes intervention masking strategy (e.g. for Table 7 ablation)."""
        self.counterfactual.set_strategy(strategy)

    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CausalFall.
        
        Args:
            x: Tensor of shape (B, C, delta)
            
        Returns:
            dict containing:
                'y_o': Original factual prediction (B, 2)
                'y_causal': Debiased causal prediction (B, 2)
                'y_cf': Counterfactual prediction SD(F_Cf) (B, 2)
                'st': Spatio-temporal features (B, C, delta)
                'c': Causal features (B, C, delta)
                'f_co': Combined joint feature (B, C, delta)
        """
        B, C, D = x.shape
        
        # Step 1: STAE feature extraction
        st = self.stae(x)  # (B, C, delta)
        
        # Step 2: CEE causal feature extraction
        if self.use_cee:
            c = self.cee(x)  # (B, C, delta)
        else:
            c = torch.zeros_like(st)
            
        # Step 3: Factual prediction through State Decoder [Eq. 12-16]
        # F_Co = ST + C + X
        f_co = st + c + x
        y_o = self.state_decoder(st, c, x)  # (B, 2)
        
        # Step 4: Counterfactual Intervention & Debiasing [Eq. 17-19]
        if self.use_counterfactual:
            # Apply intervention: [X = X']_do
            x_do = self.counterfactual(x)  # (B, C, delta)
            
            st_cf = self.stae(x_do)
            if self.use_cee:
                c_cf = self.cee(x_do)
            else:
                c_cf = torch.zeros_like(st_cf)
                
            # F_Cf = STAE([X=X']_do) + CEE([X=X']_do) [Eq. 18]
            f_cf = st_cf + c_cf + x_do
            
            # Counterfactual prediction through State Decoder: SD(F_Cf)
            # Pass f_cf through SD linear reduction, GAP, L2, softmax
            y_cf = torch.softmax(self.state_decoder.forward_features(f_cf), dim=-1)
            
            # Y_Causal = Y_o - SD(F_Cf) [Eq. 19]
            y_causal_raw = y_o - y_cf
            # Normalize to valid probability distribution for evaluation
            y_causal = torch.softmax(y_causal_raw, dim=-1)
        else:
            y_cf = torch.zeros_like(y_o)
            y_causal = y_o
            
        return {
            'y_o': y_o,
            'y_causal': y_causal,
            'y_cf': y_cf,
            'st': st,
            'c': c,
            'f_co': f_co
        }
