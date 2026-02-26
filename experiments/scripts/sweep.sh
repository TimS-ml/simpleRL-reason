#!/bin/bash
# =============================================================================
# sweep.sh — Hyperparameter sweep orchestrator
#
# Runs multiple experiments with different parameter values.
# Dispatches to train.sh, one experiment per GPU from the specified GPU list.
#
# Usage:
#   # From sweep config file
#   bash experiments/scripts/sweep.sh \
#       --base_config experiments/configs/base/grpo_qwen3_1.7b.yaml \
#       --sweep_config experiments/configs/sweeps/lr_sweep.yaml \
#       --gpus 0,1
#
#   # From inline values
#   bash experiments/scripts/sweep.sh \
#       --base_config experiments/configs/base/grpo_qwen3_1.7b.yaml \
#       --sweep_param actor.optim.lr \
#       --sweep_values "5e-7,1e-6,2e-6" \
#       --gpus 0,1
#
# Required:
#   --base_config PATH          Path to base YAML config
#
# Sweep source (mutually exclusive):
#   --sweep_config PATH         Path to sweep YAML config
#   --sweep_param KEY           Parameter to sweep (inline mode)
#   --sweep_values VALUES       Comma-separated values (inline mode)
#
# Optional:
#   --gpus IDS                  Comma-separated GPU IDs (default: "0")
#   --output_dir PATH           Checkpoint output directory (default: "./checkpoints")
#   --override KEY=VALUE        Extra overrides passed to every experiment (repeatable)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults ----
BASE_CONFIG=""
SWEEP_CONFIG=""
SWEEP_PARAM=""
SWEEP_VALUES=""
GPUS="0"
OUTPUT_DIR="./checkpoints"
EXTRA_OVERRIDES=()
HAS_EXTRA_OVERRIDES=false

# ---- Parse CLI arguments ----
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --base_config)
            BASE_CONFIG="$2"; shift 2 ;;
        --sweep_config)
            SWEEP_CONFIG="$2"; shift 2 ;;
        --sweep_param)
            SWEEP_PARAM="$2"; shift 2 ;;
        --sweep_values)
            SWEEP_VALUES="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --override)
            EXTRA_OVERRIDES+=("$2")
            HAS_EXTRA_OVERRIDES=true
            shift 2 ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            exit 1 ;;
    esac
done

# ---- Validate required args ----
if [[ -z "$BASE_CONFIG" ]]; then
    echo "ERROR: --base_config is required" >&2
    echo "Usage: $0 --base_config PATH {--sweep_config PATH | --sweep_param KEY --sweep_values VALUES} [--gpus IDS] [--output_dir PATH] [--override KEY=VALUE ...]" >&2
    exit 1
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "ERROR: Base config file not found: $BASE_CONFIG" >&2
    exit 1
fi

# Validate mutually exclusive sweep sources
if [[ -n "$SWEEP_CONFIG" ]] && { [[ -n "$SWEEP_PARAM" ]] || [[ -n "$SWEEP_VALUES" ]]; }; then
    echo "ERROR: --sweep_config is mutually exclusive with --sweep_param/--sweep_values" >&2
    exit 1
fi

# ---- Build override list ----
OVERRIDE_LIST=()
SWEEP_NAME=""

if [[ -n "$SWEEP_CONFIG" ]]; then
    # Validate sweep config exists
    if [[ ! -f "$SWEEP_CONFIG" ]]; then
        echo "ERROR: Sweep config file not found: $SWEEP_CONFIG" >&2
        exit 1
    fi

    # Extract sweep_name from YAML (use python for reliable YAML parsing)
    SWEEP_NAME=$(python3 -c "
import yaml, sys
with open('$SWEEP_CONFIG') as f:
    d = yaml.safe_load(f)
print(d.get('sweep_name', d.get('sweep_param', 'sweep')))
")

    # Parse sweep config via parse_config.py — outputs one param=value per line
    while IFS= read -r line; do
        [[ -n "$line" ]] && OVERRIDE_LIST+=("$line")
    done < <(python3 "$SCRIPT_DIR/parse_config.py" --sweep "$SWEEP_CONFIG")

elif [[ -n "$SWEEP_PARAM" ]] && [[ -n "$SWEEP_VALUES" ]]; then
    SWEEP_NAME="$SWEEP_PARAM"
    IFS=',' read -ra VALUES <<< "$SWEEP_VALUES"
    for v in "${VALUES[@]}"; do
        OVERRIDE_LIST+=("${SWEEP_PARAM}=${v}")
    done
else
    echo "ERROR: Either --sweep_config or (--sweep_param + --sweep_values) is required" >&2
    exit 1
fi

NUM_EXPERIMENTS=${#OVERRIDE_LIST[@]}

if [[ "$NUM_EXPERIMENTS" -eq 0 ]]; then
    echo "ERROR: No experiments to run (override list is empty)" >&2
    exit 1
fi

# ---- Parse GPU list ----
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_AVAILABLE_GPUS=${#GPU_ARRAY[@]}

# ---- Print summary ----
echo "============================================="
echo "Sweep: $SWEEP_NAME ($NUM_EXPERIMENTS experiments)"
echo "GPUs: $GPUS ($NUM_AVAILABLE_GPUS available)"
echo "Base config: $BASE_CONFIG"
echo "Output dir: $OUTPUT_DIR"
if [[ "$HAS_EXTRA_OVERRIDES" == true ]]; then
    echo "Extra overrides: ${EXTRA_OVERRIDES[*]}"
fi
echo "============================================="
echo ""

# ---- Run experiments with GPU queue ----
# Max concurrency = number of GPUs in --gpus list.
# Each experiment gets exactly 1 GPU from the list.
# When a GPU slot frees up (its process exits), assign the next experiment to it.

PIDS=()         # All background PIDs
GPU_PIDS=()     # PID currently running on each GPU slot (indexed by slot)
EXP_NAMES=()    # Human-readable name for each experiment (indexed by launch order)

# Initialize GPU slot tracking — empty string means slot is free
for i in "${!GPU_ARRAY[@]}"; do
    GPU_PIDS[$i]=""
done

exp_idx=0
while [[ "$exp_idx" -lt "$NUM_EXPERIMENTS" ]]; do
    # Find a free GPU slot
    gpu_slot=-1
    while [[ "$gpu_slot" -eq -1 ]]; do
        for i in "${!GPU_ARRAY[@]}"; do
            pid="${GPU_PIDS[$i]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                gpu_slot=$i
                break
            fi
        done
        if [[ "$gpu_slot" -eq -1 ]]; then
            # All GPU slots busy — wait briefly and retry
            sleep 5
        fi
    done

    # Extract override and value for this experiment
    override="${OVERRIDE_LIST[$exp_idx]}"
    gpu_id="${GPU_ARRAY[$gpu_slot]}"
    value="${override#*=}"   # everything after first '='
    suffix="${SWEEP_NAME}_${value}"

    # Build train.sh override args
    TRAIN_OVERRIDE_ARGS=(--override "$override")
    if [[ "$HAS_EXTRA_OVERRIDES" == true ]]; then
        for ov in "${EXTRA_OVERRIDES[@]}"; do
            TRAIN_OVERRIDE_ARGS+=(--override "$ov")
        done
    fi

    echo "[Exp $((exp_idx + 1))/$NUM_EXPERIMENTS] GPU $gpu_id: $override (suffix: $suffix)"

    # Launch experiment in background — individual failures must NOT abort the sweep.
    # We use a subshell so that set -e from the parent does not kill us when
    # train.sh exits non-zero; the subshell's exit status is captured by wait.
    (
        bash "$SCRIPT_DIR/train.sh" \
            --config "$BASE_CONFIG" \
            --num_gpus 1 \
            --gpus "$gpu_id" \
            --output_dir "$OUTPUT_DIR" \
            --suffix "$suffix" \
            "${TRAIN_OVERRIDE_ARGS[@]}"
    ) &

    pid=$!
    PIDS+=("$pid")
    GPU_PIDS[$gpu_slot]=$pid
    EXP_NAMES+=("$override")

    exp_idx=$((exp_idx + 1))

    # Brief stagger between launches to avoid Ray init resource contention
    if [[ "$exp_idx" -lt "$NUM_EXPERIMENTS" ]]; then
        sleep 3
    fi
done

# ---- Wait for all experiments and track results ----
echo ""
echo "All $NUM_EXPERIMENTS experiments launched. Waiting for completion..."
echo ""

failed=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  DONE:   ${EXP_NAMES[$i]}"
    else
        echo "  FAILED: ${EXP_NAMES[$i]}"
        ((failed++)) || true
    fi
done

# ---- Print final summary ----
echo ""
echo "============================================="
echo "Sweep complete: $SWEEP_NAME"
echo "  Total:  $NUM_EXPERIMENTS"
echo "  Passed: $((NUM_EXPERIMENTS - failed))"
echo "  Failed: $failed"
echo "  Logs:   experiments/logs/"
echo "============================================="

# Exit code = number of failed experiments (0 if all succeed)
exit "$failed"
