# CausalFall Replication Audit

Primary source: `reports/_paper_extract.txt`, extracted from the CausalFall base paper.

## Protocol status

| Parameter | Paper value/evidence | Implementation | Status | Notes |
|---|---|---|---|---|
| Sampling | 25 Hz for all datasets; SisFall 200 Hz and KFall 100 Hz are stated | `resample_poly`: 200->25 and 100->25 | PAPER-EXACT | Anti-aliased polyphase resampling |
| Impact | `argmax(SMV)`, SMV from accelerometer XYZ | `compute_smv` and `detect_impact_time` | PAPER-EXACT | Gyro/orientation are not used |
| Prediction horizon | theta=15 | Configurable, default 15 | PAPER-EXACT | 600 ms at 25 Hz |
| Prediction window | Section 5.2.3 uses delta=72; methodology earlier says 75 | delta=72 | PAPER-EXACT | Paper has an internal delta discrepancy; experimental value is used |
| Falling interval | `[t_impact-theta-delta, t_impact-theta)` | Exact half-open slice | PAPER-EXACT | 72 samples |
| Impact interval | CEE uses 72 samples forward from impact | `[t_impact, t_impact+delta)` | PAPER-INFERRED | Paper describes impact/post-fall stage; exact padding is not stated |
| ADL extraction | Not specified | One peak-centered window per ADL file | REPLICATION-ASSUMPTION | Configurable strategy remains needed |
| Split | Paper says datasets are randomly divided; unit is not specified | Primary file/trial split; window and subject modes available | PAPER-INFERRED | File/trial is the closest defensible interpretation; no exact claim |
| Validation | Not described | 64/16/20 train/validation/test split | REPLICATION-ASSUMPTION | Validation is required to avoid test-based checkpoint selection |
| Normalization | Not clearly specified | Train-only channel z-score, configurable off | REPLICATION-ASSUMPTION | Test statistics are never used |
| Class balance | No balancing method stated | Natural class distribution; no weighting/oversampling | PAPER-INFERRED | Counts are logged per split |
| CEE data | Offline GC over falling -> impact stages | Training fall windows and impact windows only | PAPER-INFERRED | Exact sample provenance in paper is not numerically specified |
| Granger | VAR and log residual variance ratio; retain M_GC > 1 | OLS restricted/unrestricted log variance ratio; threshold 1 | PAPER-EXACT | Lag value is not specified numerically |
| Granger lag | Orders generally do not exceed theta=15 | Configurable 5/10/15; primary 5 | REPLICATION-ASSUMPTION | Previous YAML/runtime mismatch was removed |
| CEE scale | `W_Scale` is a scaling parameter | Learnable per-channel scale initialized to one | REPLICATION-ASSUMPTION | Paper does not define scalar/vector or learning details |
| PE | Sin/cos equation with omega based on delta | Existing implementation uses channel-sized PE | CURRENT-IMPLEMENTATION | Dimension interpretation is ambiguous and requires paper-figure confirmation |
| Attention | Divide QK^T by sqrt(delta); H=9 | 72 embedding, 9 heads, scale sqrt(72) | PAPER-INFERRED | Head dimension is 8 |
| STAE | 3 MHA layers, feedforward=16 | 3 layers, feedforward=16 | PAPER-EXACT | Architecture unchanged |
| State decoder | Combine ST, C, X; paper reports BxCxdelta despite concat symbol | Elementwise `ST + C + X`, then Conv1d C->Gamma | CURRENT-IMPLEMENTATION | Paper's concat/shape notation is internally ambiguous; not changed silently |
| Gamma | Gamma < C, no numerical value stated | Gamma=4 | REPLICATION-ASSUMPTION | Exposed as configuration |
| Counterfactual | Zero first half of falling interval | First 36 of 72 samples zeroed | PAPER-EXACT | Second half unchanged |
| Debiasing | `Y_causal = Y_o - SD(F_Cf)` | Subtraction followed by softmax for usable probabilities | PAPER-INFERRED | Paper does not clarify logits versus probabilities |
| Loss | Decoder CE plus counterfactual L2 | CE on factual output plus MSE factual/causal | CURRENT-IMPLEMENTATION | Exact paper notation is dimensionally ambiguous; preserve architecture pending clarification |
| Optimizer | Adam, lr=1e-3, batch=128, epochs=500 | Same defaults | PAPER-EXACT | |
| Metrics | Accuracy, precision, recall, F1 | Fall-class sklearn metrics plus confusion matrix | PAPER-EXACT | No macro averaging |

## Current dataset counts

| Dataset | Total windows | Falls | ADLs | Fall ratio |
|---|---:|---:|---:|---:|
| SisFall | 4505 | 1798 | 2707 | 39.91% |
| KFall | 5075 | 2312 | 2763 | 45.56% |

These counts are from the current cache and differ from the paper's KFall class description (2729 ADLs and 2346 falls), although the total is also 5075. This is a dataset/protocol discrepancy, not a neural-network issue.

## Critical reproduction risks

1. The paper's phrase "randomly divides datasets" does not define a split unit. The primary file/trial split is defensible but not paper-exact.
2. The paper does not define ADL window extraction. The current peak-centered one-window-per-file policy is an explicit assumption.
3. The paper does not define a validation protocol. The implementation now selects checkpoints on validation F1 and evaluates test once.
4. The paper does not specify a numerical Granger lag. Primary lag 5 is an assumption; 10 and 15 are controlled sensitivity values.
5. Positional encoding and `ST ⊕ C ⊕ X` dimensions are ambiguous in the paper. The existing architecture is preserved rather than silently redesigned.
6. No claim of perfect reproduction is justified without the authors' preprocessing/split code.
