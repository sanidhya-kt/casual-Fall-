# CausalFall: Fall Prediction Wearing Motion Sensors from a Causal Perspective

Faithful experimental replication of the paper:
> **CausalFall: Fall prediction wearing motion sensors from a causal perspective**
> Guorui Liao, Haoyu Xie, Jiameng Li, Jun Liao, Shu Wang, Xiurong Liang, Li Liu
> *Expert Systems With Applications 300 (2026) 130218*

---

## Directory Structure
```
CausalFall_Replication/
├── config/
│   ├── default.yaml           # Global default parameters
│   ├── sisfall.yaml           # SisFall configuration (200Hz -> 25Hz)
│   └── kfall.yaml             # KFall configuration (100Hz -> 25Hz)
├── data/
│   ├── dataset.py             # PyTorch Dataset and train/test stratified split
│   ├── sisfall_loader.py      # SisFall dataset parser and 25Hz resampler
│   └── kfall_loader.py        # KFall dataset parser and 25Hz resampler
├── preprocessing/
│   ├── impact_detector.py     # SMV computation (Eq. 1) and peak impact detection
│   ├── resampler.py           # Anti-aliased decimation to 25Hz
│   ├── windowing.py           # Phase 2 falling and impact window slicing (Eq. 17)
│   └── normalizer.py          # Channel-wise Z-score standard scaling
├── causal/
│   └── granger_causality.py   # Bivariate VAR Granger causality & channel weighting (Eq. 5-11)
├── models/
│   ├── stae.py                # Spatio-Temporal Attention Encoder (STAE, 3 MHA layers, H=9)
│   ├── cee.py                 # Causal Effect Encoder (CEE, Eq. 11)
│   ├── state_decoder.py       # State Decoder (SD, Combine-Reduction-Decoding, GAP, Eq. 12-16)
│   ├── counterfactual.py      # Counterfactual Intervention (Zero-mask, Eq. 17)
│   └── causalfall.py          # Complete CausalFall end-to-end model
├── baselines/
│   ├── cnn.py                 # CNN baseline (Zhang & Zhu, 2018)
│   ├── lstm.py                # LSTM & Bi-LSTM baselines (Musci et al., 2020; Mubibya et al., 2023)
│   └── conv_gcn_transformer.py# ConvLSTM, GCN-LSTM, Transformer baselines
├── training/
│   ├── loss.py                # L_Decoder (CE) + L_Counter (L2) = L_Global (Eq. 20-23)
│   └── trainer.py             # Adam optimizer, lr=1e-3, 500 epochs training loop
├── evaluation/
│   └── metrics.py             # Accuracy, Precision, Recall, F1-Score
├── experiments/
│   ├── run_main.py            # Main experiment execution
│   ├── run_baselines.py       # All 6 baselines evaluation
│   └── run_ablations.py       # Table 5 & Table 7 ablation studies
├── tests/                     # Comprehensive unit and integration test suite
└── checkpoints/               # Saved model weights
```

---

## Paper Specifications vs Implementation Mapping

| Component | Paper Specification | Implementation | Match? |
| :--- | :--- | :--- | :--- |
| **Datasets** | SisFall, KFall, SlowFall | SisFall, KFall (SlowFall unavailable) | Faithful |
| **Sampling Frequency** | 25 Hz for all datasets | Resampled to 25 Hz (`scipy.signal.resample_poly`) | Exact |
| **Impact Detection** | $SMV = \sqrt{A_x^2 + A_y^2 + A_z^2}$, $t_{impact} = \arg\max(SMV)$ | `preprocessing.impact_detector.compute_smv` | Exact |
| **Prediction Horizon $\vartheta$** | 15 samples ($600\text{ ms}$) | $\vartheta = 15$ | Exact |
| **Window Length $\delta$** | $\delta = 72$ (from Sec 5.2.3 experimental setup) | $\delta = 72$ | Exact |
| **STAE Layers** | 3 MHA layers | 3 layers (`models.stae.STAELayer`) | Exact |
| **Attention Heads** | $H = 9$ | `num_heads = 9` | Exact |
| **FFN Dimension** | `dim_feedforward = 16` | `dim_feedforward = 16` | Exact |
| **Attention Scaling** | Scaled by $\sqrt{\delta}$ | `scores / sqrt(delta)` | Exact |
| **Causal Method** | Granger Causality (VAR OLS, log variance ratio, threshold > 1.0) | `causal.granger_causality.compute_causal_matrix` | Exact |
| **Global Causal Feature** | Precomputed offline, Eq. (11) normalized channel weights | `models.cee.CEE` | Exact |
| **State Decoder** | Combine $ST \oplus C \oplus X$, Linear $L_1$ to $\Gamma$, GAP, $L_2$, Softmax | `models.state_decoder.StateDecoder` | Exact |
| **Counterfactual Intervention** | Zero-mask first half $[0, \delta/2]$, Eq. (17) | `models.counterfactual.CounterfactualIntervention` | Exact |
| **Debiasing** | $Y_{Causal} = Y_o - SD(F_{Cf})$, Eq. (19) | `models.causalfall.CausalFall` | Exact |
| **Loss** | $L_{Global} = L_{Decoder} + L_{Counter}$, Eq. (20-23) | `training.loss.CausalFallLoss` | Exact |
| **Optimizer** | Adam, learning rate = $1 \times 10^{-3}$ | `torch.optim.Adam(lr=1e-3)` | Exact |
| **Batch Size** | 128 | `batch_size = 128` | Exact |
| **Epochs** | 500 | `epochs = 500` | Exact |

---

## How to Run Experiments

### 1. Run Unit Tests
```bash
python3 tests/test_preprocessing.py
python3 tests/test_stae.py
python3 tests/test_cee.py
python3 tests/test_state_decoder.py
python3 tests/test_counterfactual.py
python3 tests/test_loss.py
python3 tests/test_causalfall_integration.py
python3 tests/test_baselines.py
```

### 2. Run CausalFall on KFall / SisFall
```bash
python3 experiments/run_main.py --config config/kfall.yaml --data_dir /path/to/kfall
python3 experiments/run_main.py --config config/sisfall.yaml --data_dir /path/to/sisfall
```

### 3. Run Baselines
```bash
python3 experiments/run_baselines.py --config config/kfall.yaml --data_dir /path/to/kfall
```

### 4. Run Ablation Studies (Table 5 & Table 7)
```bash
python3 experiments/run_ablations.py --config config/kfall.yaml --data_dir /path/to/kfall
```
