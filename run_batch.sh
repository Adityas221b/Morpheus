#!/bin/bash
# Batch launcher for VLM jailbreak experiments
# Runs multiple models in parallel on 80GB VRAM server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
BATCH="${1:-all}"
ALGORITHM="${2:-ucb1}"
RUNS="${3:-1}"

echo "=== VLM Jailbreak Batch Runner ==="
echo "Batch: $BATCH"
echo "Algorithm: $ALGORITHM"
echo "Runs: $RUNS"
echo ""

# Define batches
declare -A BATCHES
BATCHES[small]="qwen3-vl-4b kimi-vl-3b"
BATCHES[medium]="llava-16-7b glm4v-9b qwen3-vl-8b"
BATCHES[large]="llama32-vision-11b gemma3-27b"
BATCHES[xlarge]="qwen3-vl-32b"
BATCHES[xxlarge]="qwen3-vl-72b internvl25-78b llama32-vision-90b molmo-72b"

# Function to run a single model
run_model() {
    local model=$1
    local log_file="logs/batch_${model}.log"
    echo "Starting $model (log: $log_file)"
    python3 run_vlm_hf.py --model "$model" --algorithm "$ALGORITHM" --runs "$RUNS" > "$log_file" 2>&1 &
    echo "  PID: $!"
}

# Function to wait for batch to complete
wait_batch() {
    local batch_name=$1
    echo ""
    echo "Waiting for $batch_name batch to complete..."
    wait
    echo "$batch_name batch complete!"
}

# Run batches
case $BATCH in
    small)
        for model in ${BATCHES[small]}; do
            run_model "$model"
        done
        wait_batch "small"
        ;;
    medium)
        for model in ${BATCHES[medium]}; do
            run_model "$model"
        done
        wait_batch "medium"
        ;;
    large)
        for model in ${BATCHES[large]}; do
            run_model "$model"
        done
        wait_batch "large"
        ;;
    xlarge)
        for model in ${BATCHES[xlarge]}; do
            run_model "$model"
        done
        wait_batch "xlarge"
        ;;
    xxlarge)
        for model in ${BATCHES[xxlarge]}; do
            run_model "$model"
        done
        wait_batch "xxlarge"
        ;;
    all)
        echo "Running all batches sequentially..."
        for batch in small medium large xlarge xxlarge; do
            echo ""
            echo "=== Starting $batch batch ==="
            for model in ${BATCHES[$batch]}; do
                run_model "$model"
            done
            wait_batch "$batch"
        done
        ;;
    *)
        echo "Unknown batch: $BATCH"
        echo "Available batches: small, medium, large, xlarge, xxlarge, all"
        exit 1
        ;;
esac

echo ""
echo "=== All batches complete ==="
echo "Results in: logs/vlm_hf_analysis.json"
echo "Per-model logs in: logs/batch_*.log"
