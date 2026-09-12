# CausalFall Tensor Shapes

The project uses channel-first time-series tensors throughout the model.

| Stage | Shape | Meaning |
|---|---|---|
| Raw/resampled sample | `[T, 9]` | Time by sensor channels |
| Falling or impact window | `[C, delta]` | Channels by 72 time samples |
| Batch input `X` | `[B, C, delta]` | `C=9`, `delta=72` |
| Positional embedding | `[1, C, delta]` | Added to `X` by the current implementation |
| STAE Q/K/V | `[B, H, C, head_dim]` | `H=9`, `head_dim=8` for embed dimension 72 |
| Attention scores | `[B, H, C, C]` | Channel-token attention |
| STAE output `ST` | `[B, C, delta]` | Spatio-temporal features |
| CEE output `C` | `[B, C, delta]` | Channel-weighted causal features |
| Combined decoder input | `[B, C, delta]` | Current implementation uses `ST + C + X` |
| Reduced decoder feature | `[B, Gamma, delta]` | `Gamma=4` current assumption |
| GAP feature | `[B, 1]` | Mean over Gamma and delta |
| Decoder output | `[B, 2]` | Two-class softmax |
| Counterfactual input | `[B, C, delta]` | First 36 time samples zeroed for primary mask |
| Causal prediction | `[B, 2]` | Factual minus counterfactual, normalized for prediction |
| Causal matrix | `[C, C]` | Falling channel to impact channel |
| Channel weights | `[C]` | Row sums normalized by total causal mass |

## Shape caveat

The paper writes `F_Co = ST ⊕ C ⊕ X` while also assigning `F_Co` shape `[B,C,delta]`. The current repository resolves this as elementwise addition. A literal channel concatenation would produce `[B,3C,delta]` and require a different decoder input layer, so it is not introduced without stronger primary-source evidence.
