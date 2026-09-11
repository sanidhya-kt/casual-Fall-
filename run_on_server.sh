#!/bin/bash
# ==============================================================================
# CausalFall Server GPU Execution Script
# Reproduces:
#   1. Main CausalFall experiment (500 epochs, Adam, lr=1e-3, batch=128)
#   2. All 6 Baseline models (CNN, LSTM, Bi-LSTM, ConvLSTM, GCN-LSTM, Transformer)
#   3. CEE Ablation (with & without CEE)
#   4. Counterfactual Masking Ablations (Zero, Mean, Random, No-Intervention)
# ==============================================================================

set -e

# Default to KFall or pass sisfall as first argument
DATASET=${1:-kfall}
DEVICE=${2:-auto}

echo "=========================================================="
echo "Starting CausalFall Replication on Dataset: $DATASET"
echo "Device: $DEVICE"
echo "=========================================================="

# 1. Build preprocessed cache if not already present
if [ ! -f "./data/cache/${DATASET}_preprocessed.npz" ]; then
    echo "Preprocessed cache not found. Building cache from raw data..."
    python3 preprocessing/build_cache.py --dataset "$DATASET"
fi

# 2. Run Main CausalFall Experiment (500 epochs)
echo ""
echo ">>> [1/3] Running CausalFall Main Model..."
python3 experiments/run_main.py --config "config/${DATASET}.yaml"

# 3. Run Baseline Comparisons (Table 4)
echo ""
echo ">>> [2/3] Running All 6 Baselines..."
python3 experiments/run_baselines.py --config "config/${DATASET}.yaml"

# 4. Run Ablation Studies (Tables 5 & 7)
echo ""
echo ">>> [3/3] Running Ablations (CEE & Counterfactual Masking)..."
python3 experiments/run_ablations.py --config "config/${DATASET}.yaml"

echo ""
echo "=========================================================="
echo "All Experiments Completed Successfully!"
echo "Checkpoints saved in ./checkpoints/"
echo "=========================================================="
