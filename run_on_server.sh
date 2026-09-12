#!/usr/bin/env bash
# CausalFall GPU runner for SisFall and KFall.
#
# Safe defaults:
#   ./run_on_server.sh                 # both datasets, sequentially
#   ./run_on_server.sh --parallel      # both datasets, at most two jobs
#   ./run_on_server.sh sisfall         # one dataset

set -uo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR" || exit 1

DATASETS=(sisfall kfall)
PARALLEL=0
MAIN_ONLY=0
EPOCHS=""
SEED=""
MAX_LAG=""
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/server_runs}"
JOB_TIMEOUT="${JOB_TIMEOUT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  ./run_on_server.sh [sisfall|kfall] [options]
  ./run_on_server.sh --parallel [options]

Options:
  --parallel       Run SisFall and KFall concurrently (maximum two jobs).
  --main-only      Run only the CausalFall main model, not baselines/ablations.
  --epochs N       Override epochs for a controlled smoke test.
  --seed N         Override the random seed.
  --max-lag N      Granger lag override: 5, 10, or 15.
  --timeout SEC    Kill an individual dataset job after SEC seconds; 0 disables it.
  --log-dir PATH   Store logs under PATH.
EOF
}

while (($# > 0)); do
    case "$1" in
        sisfall|kfall)
            DATASETS=("$1")
            shift
            ;;
        --parallel)
            PARALLEL=1
            shift
            ;;
        --main-only)
            MAIN_ONLY=1
            shift
            ;;
        --epochs|--seed|--max-lag|--timeout|--log-dir)
            if (($# < 2)); then
                echo "Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            case "$1" in
                --epochs) EPOCHS="$2" ;;
                --seed) SEED="$2" ;;
                --max-lag) MAX_LAG="$2" ;;
                --timeout) JOB_TIMEOUT="$2" ;;
                --log-dir) LOG_DIR="$2" ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 127
fi

mkdir -p "$LOG_DIR"
RUN_ID=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$LOG_DIR/summary_${RUN_ID}.txt"
RESULT_DIR="$LOG_DIR/$RUN_ID"
mkdir -p "$RESULT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU status before launch:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
else
    echo "WARNING: nvidia-smi is unavailable; GPU preflight was skipped." >&2
fi

build_args() {
    local dataset="$1"
    RUN_ARGS=("--config" "$ROOT_DIR/config/${dataset}.yaml")
    [[ -n "$EPOCHS" ]] && RUN_ARGS+=("--epochs" "$EPOCHS")
    [[ -n "$SEED" ]] && RUN_ARGS+=("--seed" "$SEED")
    [[ -n "$MAX_LAG" ]] && RUN_ARGS+=("--max-lag" "$MAX_LAG")
}

run_python_step() {
    local label="$1"
    shift
    echo "[$(date '+%F %T')] START $label"
    if (( JOB_TIMEOUT > 0 )) && command -v timeout >/dev/null 2>&1; then
        timeout --signal=TERM --kill-after=60 "$JOB_TIMEOUT" "$PYTHON_BIN" -u "$@"
    else
        "$PYTHON_BIN" -u "$@"
    fi
    local code=$?
    if (( code != 0 )); then
        echo "[$(date '+%F %T')] FAIL $label (exit=$code)" >&2
        return "$code"
    fi
    echo "[$(date '+%F %T')] DONE $label"
    return 0
}

run_dataset() {
    local dataset="$1"
    local log_file="$RESULT_DIR/${dataset}.log"
    local code=0

    (
        set -uo pipefail
        exec > >(tee -a "$log_file") 2>&1
        trap 'echo "[$(date "+%F %T")] INTERRUPTED" >&2; exit 130' INT TERM

        echo "=========================================================="
        echo "Starting CausalFall replication: $dataset"
        echo "Host: $(hostname)"
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
        echo "Main only: $MAIN_ONLY | Parallel batch: $PARALLEL"
        echo "=========================================================="

        export PYTHONUNBUFFERED=1
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

        if [[ ! -f "$ROOT_DIR/data/cache/${dataset}_preprocessed.npz" ]]; then
            run_python_step "$dataset cache build" preprocessing/build_cache.py --dataset "$dataset" || exit $?
        fi

        build_args "$dataset"
        run_python_step "$dataset main model" experiments/run_main.py "${RUN_ARGS[@]}" || exit $?

        if (( MAIN_ONLY == 0 )); then
            run_python_step "$dataset baselines" experiments/run_baselines.py "${RUN_ARGS[@]}" || exit $?
            run_python_step "$dataset ablations" experiments/run_ablations.py "${RUN_ARGS[@]}" || exit $?
        fi

        echo "[$(date '+%F %T')] COMPLETED $dataset"
    )
    code=$?

    if (( code == 0 )); then
        printf '%s PASS\n' "$dataset" >> "$SUMMARY_FILE"
    else
        printf '%s FAIL exit=%s log=%s\n' "$dataset" "$code" "$log_file" >> "$SUMMARY_FILE"
    fi
    return "$code"
}

: > "$SUMMARY_FILE"
PIDS=()
if (( PARALLEL == 1 )) && ((${#DATASETS[@]} > 1)); then
    echo "Launching ${#DATASETS[@]} isolated dataset jobs in parallel."
    for dataset in "${DATASETS[@]}"; do
        run_dataset "$dataset" &
        PIDS+=("$!")
    done
    overall_code=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || overall_code=1
    done
else
    overall_code=0
    for dataset in "${DATASETS[@]}"; do
        run_dataset "$dataset" || overall_code=1
    done
fi

echo
echo "==================== RUN SUMMARY ========================="
cat "$SUMMARY_FILE"
echo "Logs: $RESULT_DIR"
echo "=========================================================="
exit "$overall_code"
