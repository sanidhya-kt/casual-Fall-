import subprocess
import os

commits = [
    (
        ["requirements.txt", ".gitignore", "config/"],
        "build: setup project configuration, requirements, and gitignore"
    ),
    (
        ["preprocessing/impact_detector.py", "tests/test_preprocessing.py"],
        "feat(preprocessing): implement SMV computation and peak impact detection (Eq. 1)"
    ),
    (
        ["preprocessing/resampler.py", "preprocessing/normalizer.py"],
        "feat(preprocessing): implement anti-aliased 25Hz resampling and channel normalizer"
    ),
    (
        ["preprocessing/windowing.py"],
        "feat(preprocessing): implement Phase 2 falling/impact window slicing and ADL extraction"
    ),
    (
        ["data/sisfall_loader.py", "data/kfall_loader.py"],
        "feat(data): implement raw dataset loaders for SisFall (200Hz) and KFall (100Hz)"
    ),
    (
        ["data/dataset.py"],
        "feat(data): implement PyTorch FallDataset with stratified splitting and caching support"
    ),
    (
        ["models/stae.py", "tests/test_stae.py"],
        "feat(models): implement Spatio-Temporal Attention Encoder (STAE) with sinusoidal PE"
    ),
    (
        ["causal/granger_causality.py"],
        "feat(causal): implement Granger Causality analysis and bivariate VAR OLS (Eq. 5-10)"
    ),
    (
        ["models/cee.py", "tests/test_cee.py"],
        "feat(models): implement Causal Effect Encoder (CEE) with normalized channel weights (Eq. 11)"
    ),
    (
        ["models/state_decoder.py", "tests/test_state_decoder.py"],
        "feat(models): implement State Decoder with Combine-Reduction-Decoding, GAP, and Softmax"
    ),
    (
        ["models/counterfactual.py", "tests/test_counterfactual.py"],
        "feat(models): implement Counterfactual Intervention module (Zero, Mean, Random, None)"
    ),
    (
        ["training/loss.py", "tests/test_loss.py"],
        "feat(training): implement Debiasing and loss functions L_Decoder, L_Counter, L_Global"
    ),
    (
        ["models/causalfall.py", "models/__init__.py", "tests/test_causalfall_integration.py"],
        "feat(models): integrate complete end-to-end CausalFall neural network architecture"
    ),
    (
        ["baselines/", "tests/test_baselines.py"],
        "feat(baselines): implement CNN, LSTM, Bi-LSTM, ConvLSTM, GCN-LSTM, and Transformer"
    ),
    (
        [
            "training/trainer.py",
            "evaluation/metrics.py",
            "tests/test_pipeline_dryrun.py",
            "tests/test_real_data_quick_check.py",
            "run_all_tests.py"
        ],
        "feat(training): implement Trainer, evaluation metrics, and comprehensive test suites"
    ),
    (
        [
            "experiments/run_main.py",
            "experiments/run_baselines.py",
            "experiments/run_ablations.py"
        ],
        "feat(experiments): implement experiment runners for main model, baselines, and ablations"
    ),
    (
        [
            "preprocessing/build_cache.py",
            "run_on_server.sh",
            "README.md",
            "checkpoints/.gitkeep",
            "results/.gitkeep",
            "datasets/documentation/",
            "datasets/processed/"
        ],
        "docs & ci: add documentation, server GPU runner script, and dataset documentation"
    ),
    (
        ["data/cache/"],
        "data: add preprocessed 25Hz compressed caches for SisFall and KFall datasets"
    )
]

print("Executing progressive commits...")
for files, msg in commits:
    for f in files:
        if os.path.exists(f):
            subprocess.run(["git", "add", f], check=True)
    # Check if anything is staged
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if staged.returncode != 0:
        res = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
        print(f"[COMMITTED] {msg}")
    else:
        print(f"[SKIPPED] No changes to commit for: {msg}")

# Final commit for any remaining untracked items
staged_any = subprocess.run(["git", "add", "."])
staged_check = subprocess.run(["git", "diff", "--cached", "--quiet"])
if staged_check.returncode != 0:
    subprocess.run(["git", "commit", "-m", "chore: finalize repository files and directory structure"])
    print("[COMMITTED] chore: finalize repository files and directory structure")

print("\nAll commits created successfully!")
