#!/bin/bash
# =============================================================================
# self_evolve.sh — Self-evolving RL training loop
#
# Runs multiple rounds of hyperparameter sweeps. After each round, selects the
# best checkpoint (by val/correctness) and uses it as the starting model for
# the next round. This is the config-driven replacement for run_self_evolve_rl.sh.
#
# Usage:
#   bash experiments/scripts/self_evolve.sh \
#       --base_config experiments/configs/base/grpo_qwen3_1.7b.yaml \
#       --sweep_config experiments/configs/sweeps/lr_sweep.yaml \
#       --gpus 0,1,2 \
#       --rounds 3 \
#       --epochs_per_round 10 \
#       --output_dir ./checkpoints/self_evolve
#
# Required:
#   --base_config PATH         Path to base YAML config
#   --sweep_config PATH        Path to sweep YAML config
#
# Optional:
#   --gpus IDS                 Comma-separated GPU IDs (default: "0")
#   --rounds N                 Number of self-evolve rounds (default: 3)
#   --epochs_per_round N       Training epochs per round (default: 10)
#   --output_dir PATH          Output directory (default: "./checkpoints/self_evolve")
# =============================================================================

set -euo pipefail

# ---- Resolve paths relative to this script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Defaults ----
BASE_CONFIG=""
SWEEP_CONFIG=""
GPUS="0"
ROUNDS=3
EPOCHS_PER_ROUND=10
OUTPUT_DIR="./checkpoints/self_evolve"

# ---- Parse CLI arguments ----
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --base_config)
            BASE_CONFIG="$2"; shift 2 ;;
        --sweep_config)
            SWEEP_CONFIG="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --rounds)
            ROUNDS="$2"; shift 2 ;;
        --epochs_per_round)
            EPOCHS_PER_ROUND="$2"; shift 2 ;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            exit 1 ;;
    esac
done

# ---- Validate required args ----
if [[ -z "$BASE_CONFIG" ]]; then
    echo "ERROR: --base_config is required" >&2
    echo "Usage: $0 --base_config PATH --sweep_config PATH [--gpus IDS] [--rounds N] [--epochs_per_round N] [--output_dir PATH]" >&2
    exit 1
fi

if [[ -z "$SWEEP_CONFIG" ]]; then
    echo "ERROR: --sweep_config is required" >&2
    echo "Usage: $0 --base_config PATH --sweep_config PATH [--gpus IDS] [--rounds N] [--epochs_per_round N] [--output_dir PATH]" >&2
    exit 1
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "ERROR: Base config file not found: $BASE_CONFIG" >&2
    exit 1
fi

if [[ ! -f "$SWEEP_CONFIG" ]]; then
    echo "ERROR: Sweep config file not found: $SWEEP_CONFIG" >&2
    exit 1
fi

# ---- Get initial model path from base config ----
PARSE_CONFIG="$SCRIPT_DIR/parse_config.py"
INITIAL_MODEL=$(python3 "$PARSE_CONFIG" "$BASE_CONFIG" --format flat \
    | grep "^actor_rollout_ref\.model\.path=" \
    | head -1 \
    | cut -d'=' -f2-)

if [[ -z "$INITIAL_MODEL" ]]; then
    echo "ERROR: Could not extract model.path from base config: $BASE_CONFIG" >&2
    exit 1
fi

CURRENT_MODEL="$INITIAL_MODEL"

# ---- Create output directory ----
mkdir -p "$OUTPUT_DIR"

# ---- Print overall summary ----
echo "============================================="
echo "Self-Evolving RL Training"
echo "============================================="
echo "Base config:       $BASE_CONFIG"
echo "Sweep config:      $SWEEP_CONFIG"
echo "Initial model:     $INITIAL_MODEL"
echo "GPUs:              $GPUS"
echo "Rounds:            $ROUNDS"
echo "Epochs per round:  $EPOCHS_PER_ROUND"
echo "Output dir:        $OUTPUT_DIR"
echo "============================================="

# ---- Track results per round ----
ROUND_RESULTS=()

# ---- Main self-evolve loop ----
for round in $(seq 1 "$ROUNDS"); do
    echo ""
    echo "===== ROUND $round / $ROUNDS ====="
    echo "Starting model: $CURRENT_MODEL"
    echo ""

    ROUND_DIR="$OUTPUT_DIR/round${round}"

    # Build sweep.sh command — sweep failures should NOT abort the loop
    SWEEP_CMD=(
        bash "$SCRIPT_DIR/sweep.sh"
        --base_config "$BASE_CONFIG"
        --sweep_config "$SWEEP_CONFIG"
        --gpus "$GPUS"
        --output_dir "$ROUND_DIR"
        --override "trainer.total_epochs=$EPOCHS_PER_ROUND"
    )

    # For round 2+, override model path with the best checkpoint from prior round
    if [[ "$round" -gt 1 ]]; then
        SWEEP_CMD+=(--override "model.path=$CURRENT_MODEL")
    fi

    echo "Running sweep for round $round..."
    if "${SWEEP_CMD[@]}"; then
        echo "Sweep for round $round completed successfully."
    else
        echo "WARNING: sweep.sh exited with non-zero status for round $round. Some experiments may have failed."
    fi

    # ---- Select best checkpoint from this round ----
    echo ""
    echo "--- Selecting best checkpoint from round $round ---"

    LOG_DIR="$PROJECT_ROOT/experiments/logs"
    best_score=0
    best_variant=""
    best_ckpt=""

    # Scan log files in experiments/logs/ for this round
    # Log files are named like: <algorithm>_<model>_<gpus>gpu_lr<lr>_<timestamp>_<sweep_suffix>.log
    # The round's experiments are in ROUND_DIR, so we look for logs whose corresponding
    # checkpoint directories are under ROUND_DIR
    if [[ -d "$LOG_DIR" ]]; then
        for log_file in "$LOG_DIR"/*.log; do
            [[ -f "$log_file" ]] || continue

            # Check if this log's checkpoint dir is under our ROUND_DIR
            # The log file basename (minus .log) is the RUN_NAME, and the checkpoint
            # dir is OUTPUT_DIR/RUN_NAME (which for sweeps = ROUND_DIR/RUN_NAME)
            log_basename="$(basename "$log_file" .log)"
            ckpt_base="$ROUND_DIR/$log_basename"

            # Only consider logs whose checkpoint dir exists under this round
            [[ -d "$ckpt_base" ]] || continue

            # Extract best val/correctness score from this log
            score=$(grep -oP "val/correctness['\"]?\s*[:=]\s*\K[0-9.]+" "$log_file" 2>/dev/null | sort -rn | head -1 || true)

            if [[ -z "$score" ]]; then
                echo "  $log_basename: no val/correctness score found"
                continue
            fi

            echo "  $log_basename: score=$score"

            # Compare scores (using awk for float comparison)
            is_better=$(awk "BEGIN {print ($score > $best_score) ? 1 : 0}")
            if [[ "$is_better" -eq 1 ]]; then
                best_score="$score"
                best_variant="$log_basename"

                # Find the latest global_step_* checkpoint subdirectory
                latest_ckpt=$(ls -td "${ckpt_base}"/global_step_* 2>/dev/null | head -1 || true)
                if [[ -n "$latest_ckpt" ]] && [[ -d "$latest_ckpt" ]]; then
                    best_ckpt="$latest_ckpt"
                fi
            fi
        done
    fi

    # ---- Round summary ----
    echo ""
    if [[ -n "$best_ckpt" ]] && [[ -d "$best_ckpt" ]]; then
        echo "Round $round best: $best_variant (score: $best_score)"
        echo "Checkpoint: $best_ckpt"
        CURRENT_MODEL="$best_ckpt"
        ROUND_RESULTS+=("Round $round: $best_variant — score=$best_score — $best_ckpt")
    else
        echo "WARNING: No valid checkpoint found for round $round."
        echo "  Continuing with current model: $CURRENT_MODEL"
        ROUND_RESULTS+=("Round $round: no valid checkpoint found — kept $CURRENT_MODEL")
    fi

    echo "===== END ROUND $round ====="
done

# ---- Final summary ----
echo ""
echo "============================================="
echo "Self-Evolving RL Complete"
echo "============================================="
echo ""
echo "Round-by-round results:"
for result in "${ROUND_RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "Final model: $CURRENT_MODEL"
echo "============================================="
