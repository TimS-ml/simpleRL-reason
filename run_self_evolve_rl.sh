#!/bin/bash
# =============================================================================
# Self-Evolving RL: Run 6 independent GRPO experiments on 6x RTX 4090
#
# Each GPU runs an independent experiment with different hyperparameters
# for diversity. After N epochs, the best checkpoint is selected and used
# as the starting point for the next round of training.
#
# Self-Evolve Loop:
#   Round 1: Train 6 variants with different HP from base model
#   Round 2: Pick best checkpoint, continue training with new HP sweep
#   ...repeat...
#
# Usage:
#   bash run_self_evolve_rl.sh
#
# Required env vars:
#   MODEL_PATH  - Path to base Qwen3-4B model
#   DATA_DIR    - Path to data directory with train.parquet & test.parquet
#
# Optional env vars:
#   NUM_ROUNDS       - Number of self-evolve rounds (default: 3)
#   EPOCHS_PER_ROUND - Training epochs per round (default: 10)
#   OUTPUT_DIR       - Base output directory (default: ./checkpoints/self_evolve)
# =============================================================================

set -euo pipefail

# ---- Configuration ----
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-./data/simplelr_qwen_level3to5}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/self_evolve}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
EPOCHS_PER_ROUND="${EPOCHS_PER_ROUND:-10}"
NUM_GPUS=6

# ---- Hyperparameter variants for 6 GPUs ----
# Each GPU explores a different region of the HP space
# Format: "lr,kl_loss_coef,rollout_n,temperature,entropy_coeff"
HP_CONFIGS=(
    "1e-6,0.0001,4,0.6,0.001"    # GPU 0: baseline conservative
    "5e-7,0.0001,4,0.6,0.001"    # GPU 1: lower LR
    "1e-6,0.001,4,0.6,0.001"     # GPU 2: higher KL penalty
    "1e-6,0.0001,8,0.6,0.0005"   # GPU 3: more rollouts, lower entropy
    "2e-6,0.0001,4,0.8,0.001"    # GPU 4: higher LR + temperature
    "1e-6,0.0001,4,0.6,0.003"    # GPU 5: higher entropy exploration
)

VARIANT_NAMES=(
    "baseline"
    "low_lr"
    "high_kl"
    "more_rollouts"
    "high_lr_temp"
    "high_entropy"
)

mkdir -p "$OUTPUT_DIR"

# ---- Helper: Parse HP config string ----
parse_hp() {
    IFS=',' read -r LR KL_LOSS_COEF ROLLOUT_N TEMP ENTROPY <<< "$1"
    echo "$LR $KL_LOSS_COEF $ROLLOUT_N $TEMP $ENTROPY"
}

# ---- Helper: Extract best validation score from training log ----
get_best_score() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        # Look for validation accuracy in the log
        grep -oP "val/correctness['\"]?\s*[:=]\s*\K[0-9.]+" "$log_file" | sort -rn | head -1
    else
        echo "0"
    fi
}

# ---- Helper: Run single GPU experiment ----
run_experiment() {
    local gpu_id=$1
    local model_path=$2
    local round_num=$3
    local variant_name=$4
    local hp_config=$5
    local epochs=$6

    read -r LR KL_LOSS_COEF ROLLOUT_N TEMP ENTROPY <<< "$(parse_hp "$hp_config")"

    local run_name="round${round_num}_${variant_name}"
    local ckpt_dir="${OUTPUT_DIR}/${run_name}"
    local log_file="${ckpt_dir}.log"

    echo "[GPU $gpu_id] Starting: $run_name (lr=$LR, kl=$KL_LOSS_COEF, n=$ROLLOUT_N, temp=$TEMP, ent=$ENTROPY)"

    CUDA_VISIBLE_DEVICES=$gpu_id bash train_qwen3_4b_single_gpu.sh \
        --model_path "$model_path" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate "$LR" \
        --kl_loss_coef "$KL_LOSS_COEF" \
        --rollout_n "$ROLLOUT_N" \
        --temperature "$TEMP" \
        --entropy_coeff "$ENTROPY" \
        --total_epochs "$epochs" \
        --suffix "$run_name" \
        > "$log_file" 2>&1

    echo "[GPU $gpu_id] Finished: $run_name"
}

# ---- Main Self-Evolve Loop ----
echo "============================================="
echo "Self-Evolving RL Training"
echo "============================================="
echo "Base model:        $MODEL_PATH"
echo "Data:              $DATA_DIR"
echo "Output:            $OUTPUT_DIR"
echo "Rounds:            $NUM_ROUNDS"
echo "Epochs per round:  $EPOCHS_PER_ROUND"
echo "Num GPUs:          $NUM_GPUS"
echo "============================================="

current_model="$MODEL_PATH"

for round in $(seq 1 "$NUM_ROUNDS"); do
    echo ""
    echo "===== ROUND $round / $NUM_ROUNDS ====="
    echo "Starting model: $current_model"
    echo ""

    # Launch all 6 experiments in parallel
    pids=()
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        run_experiment "$gpu_id" "$current_model" "$round" "${VARIANT_NAMES[$gpu_id]}" "${HP_CONFIGS[$gpu_id]}" "$EPOCHS_PER_ROUND" &
        pids+=($!)
        # Stagger launches slightly to avoid resource contention during init
        sleep 5
    done

    # Wait for all experiments to finish
    echo "Waiting for all $NUM_GPUS experiments to complete..."
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "WARNING: Process $pid failed"
            ((failed++))
        fi
    done

    if [ "$failed" -eq "$NUM_GPUS" ]; then
        echo "ERROR: All experiments in round $round failed. Aborting."
        exit 1
    fi

    # ---- Select best checkpoint ----
    echo ""
    echo "--- Selecting best checkpoint from round $round ---"
    best_score=0
    best_ckpt=""

    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        variant="${VARIANT_NAMES[$gpu_id]}"
        run_name="round${round}_${variant}"
        log_file="${OUTPUT_DIR}/verl-grpo_*_${run_name}.log"

        # Find the actual log file (glob expansion)
        for f in $log_file; do
            if [ -f "$f" ]; then
                score=$(get_best_score "$f")
                score=${score:-0}
                echo "  $variant: score=$score"

                # Compare scores (using awk for float comparison)
                is_better=$(awk "BEGIN {print ($score > $best_score) ? 1 : 0}")
                if [ "$is_better" -eq 1 ]; then
                    best_score=$score
                    # Find corresponding checkpoint directory
                    ckpt_base="${f%.log}"
                    if [ -d "$ckpt_base" ]; then
                        # Find the latest checkpoint subdirectory
                        latest_ckpt=$(ls -td "${ckpt_base}"/global_step_* 2>/dev/null | head -1)
                        if [ -n "$latest_ckpt" ]; then
                            best_ckpt="$latest_ckpt"
                        fi
                    fi
                fi
            fi
        done
    done

    if [ -n "$best_ckpt" ] && [ -d "$best_ckpt" ]; then
        echo ""
        echo "Best checkpoint: $best_ckpt (score: $best_score)"
        current_model="$best_ckpt"
    else
        echo ""
        echo "WARNING: No valid checkpoint found. Continuing with current model."
        echo "  This may happen if training logs don't contain validation scores."
        echo "  Check logs in $OUTPUT_DIR for details."
    fi

    echo "===== END ROUND $round ====="
done

echo ""
echo "============================================="
echo "Self-Evolving RL Complete"
echo "Final best model: $current_model"
echo "============================================="
