#!/bin/bash
# =============================================================================
# train.sh — Universal single-experiment launcher
#
# Reads a YAML config via parse_config.py, sets up the environment, starts Ray,
# and launches verl training. Can be called standalone or by sweep.sh / self_evolve.sh.
#
# Usage:
#   experiments/scripts/train.sh --config experiments/configs/base/grpo_qwen3_1.7b.yaml
#   experiments/scripts/train.sh --config experiments/configs/base/grpo_qwen3_1.7b.yaml \
#       --num_gpus 2 --gpus 0,1 --override actor.optim.lr=5e-7
#
# Required:
#   --config PATH          Path to YAML config file
#
# Optional:
#   --num_gpus N           Number of GPUs (default: 1)
#   --gpus IDS             Comma-separated GPU IDs for CUDA_VISIBLE_DEVICES (default: "0")
#   --output_dir PATH      Checkpoint output directory (default: "./checkpoints")
#   --suffix STRING        Optional suffix appended to run name
#   --override KEY=VALUE   Override config params, repeatable (short keys)
#   --model_path PATH      Override model path (e.g. checkpoint from self_evolve.sh)
#   --dry_run              Parse config and print header, but don't start training
# =============================================================================

set -euo pipefail

# ---- Resolve paths relative to this script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Defaults ----
CONFIG=""
NUM_GPUS=1
GPUS="0"
OUTPUT_DIR="./checkpoints"
SUFFIX=""
MODEL_PATH=""
DRY_RUN=false
# Use a placeholder to handle bash set -u with empty arrays
OVERRIDES=()
HAS_OVERRIDES=false

# ---- Parse CLI arguments ----
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --num_gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --suffix)
            SUFFIX="$2"; shift 2 ;;
        --model_path)
            MODEL_PATH="$2"; shift 2 ;;
        --dry_run)
            DRY_RUN=true; shift ;;
        --override)
            OVERRIDES+=("$2")
            HAS_OVERRIDES=true
            shift 2 ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            exit 1 ;;
    esac
done

# ---- Validate required args ----
if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required" >&2
    echo "Usage: $0 --config PATH [--num_gpus N] [--gpus IDS] [--output_dir PATH] [--suffix STRING] [--override KEY=VALUE ...]" >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

# ---- Build override args for parse_config.py ----
# Always override n_gpus_per_node to match --num_gpus
PARSE_OVERRIDES=()
PARSE_OVERRIDES+=("--override" "trainer.n_gpus_per_node=$NUM_GPUS")

# If --model_path provided, override model.path
if [[ -n "$MODEL_PATH" ]]; then
    PARSE_OVERRIDES+=("--override" "model.path=$MODEL_PATH")
fi

# If multi-GPU, disable optimizer offload (enough VRAM across GPUs)
if [[ "$NUM_GPUS" -gt 1 ]]; then
    PARSE_OVERRIDES+=("--override" "actor.fsdp_config.optimizer_offload=false")
fi

# Pass through user --override args
if [[ "$HAS_OVERRIDES" == true ]]; then
    for ov in "${OVERRIDES[@]}"; do
        PARSE_OVERRIDES+=("--override" "$ov")
    done
fi

PARSE_CONFIG="$SCRIPT_DIR/parse_config.py"

# ---- Call parse_config.py: flat format (verl CLI args) ----
FLAT_OUTPUT=$(python "$PARSE_CONFIG" "$CONFIG" --format flat "${PARSE_OVERRIDES[@]}")

# ---- Call parse_config.py: wandb format (JSON reference) ----
WANDB_JSON=$(python "$PARSE_CONFIG" "$CONFIG" --format wandb "${PARSE_OVERRIDES[@]}")

# ---- Extract values from flat output for run naming ----
# Flat output has one verl_key=value per line
get_flat_value() {
    local key="$1"
    echo "$FLAT_OUTPUT" | grep "^${key}=" | head -1 | cut -d'=' -f2-
}

MODEL_PATH_RESOLVED=$(get_flat_value "actor_rollout_ref.model.path")
ALGORITHM=$(get_flat_value "algorithm.adv_estimator")
LR=$(get_flat_value "actor_rollout_ref.actor.optim.lr")

# Extract short model name from model path (last component, lowercase)
MODEL_SHORT=$(basename "$MODEL_PATH_RESOLVED" | tr '[:upper:]' '[:lower:]')

# ---- Generate run name ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${ALGORITHM}_${MODEL_SHORT}_${NUM_GPUS}gpu_lr${LR}_${TIMESTAMP}"
if [[ -n "$SUFFIX" ]]; then
    RUN_NAME="${RUN_NAME}_${SUFFIX}"
fi

# ---- Call parse_config.py: header format (log header text) ----
# Build override display list for header
HEADER_OVERRIDE_ARGS=()
if [[ "$HAS_OVERRIDES" == true ]]; then
    for ov in "${OVERRIDES[@]}"; do
        HEADER_OVERRIDE_ARGS+=("--override" "$ov")
    done
fi
# Also include the implicit overrides in the header
HEADER_OVERRIDE_ARGS+=("--override" "trainer.n_gpus_per_node=$NUM_GPUS")
if [[ -n "$MODEL_PATH" ]]; then
    HEADER_OVERRIDE_ARGS+=("--override" "model.path=$MODEL_PATH")
fi
if [[ "$NUM_GPUS" -gt 1 ]]; then
    HEADER_OVERRIDE_ARGS+=("--override" "actor.fsdp_config.optimizer_offload=false")
fi

HEADER_OUTPUT=$(python "$PARSE_CONFIG" "$CONFIG" --format header \
    --run_name "$RUN_NAME" \
    --gpus "$GPUS" \
    --num_gpus "$NUM_GPUS" \
    "${HEADER_OVERRIDE_ARGS[@]}")

# ---- Create directories ----
LOG_DIR="$PROJECT_ROOT/experiments/logs"
CKPT_DIR="${OUTPUT_DIR}/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "$LOG_DIR"
mkdir -p "$CKPT_DIR"

# ---- Write header to log file ----
echo "$HEADER_OUTPUT" > "$LOG_FILE"
echo "" >> "$LOG_FILE"

# ---- Print header to stdout ----
echo "$HEADER_OUTPUT"
echo ""

# ---- Save wandb JSON for reference ----
echo "$WANDB_JSON" > "${LOG_DIR}/${RUN_NAME}_wandb.json"

# ---- Dry run: stop here ----
if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would launch training with run name: $RUN_NAME"
    echo "[DRY RUN] Checkpoint dir: $CKPT_DIR"
    echo "[DRY RUN] Log file: $LOG_FILE"
    echo "[DRY RUN] Flat config args:"
    echo "$FLAT_OUTPUT"
    echo ""
    echo "[DRY RUN] Wandb JSON saved to: ${LOG_DIR}/${RUN_NAME}_wandb.json"
    echo "[DRY RUN] Exiting without starting Ray or training."
    exit 0
fi

# ---- Set environment ----
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES="$GPUS"

# ---- Start Ray ----
echo "Starting Ray (num_gpus=$NUM_GPUS)..."
ray stop --force 2>/dev/null || true
ray start --head --num-gpus "$NUM_GPUS" --num-cpus 8
sleep 3

# ---- Build verl command ----
# Read flat output into an array of key=value args
VERL_ARGS=()
while IFS= read -r line; do
    [[ -n "$line" ]] && VERL_ARGS+=("$line")
done <<< "$FLAT_OUTPUT"

# Override checkpoint dir and experiment name
VERL_ARGS+=("trainer.default_local_dir=$CKPT_DIR")
VERL_ARGS+=("trainer.experiment_name=$RUN_NAME")

# ---- Run training ----
echo "Launching verl training..."
echo "  Run name: $RUN_NAME"
echo "  Checkpoint dir: $CKPT_DIR"
echo "  Log file: $LOG_FILE"
echo ""

python -m verl.trainer.main_ppo "${VERL_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

# ---- Completion summary ----
echo ""
echo "============================================="
echo "Training complete!"
echo "  Run name:       $RUN_NAME"
echo "  Checkpoint dir: $CKPT_DIR"
echo "  Log file:       $LOG_FILE"
echo "============================================="
