#!/bin/bash
# =============================================================================
# run_experiments.sh — Shortcut entry point for the experiment framework
#
# Wraps experiments/scripts/{train,sweep,self_evolve}.sh with shorter syntax.
# Auto-expands config names to full paths.
#
# Usage:
#   bash run_experiments.sh train --config grpo_qwen3_1.7b --gpus 0
#   bash run_experiments.sh train --config grpo_qwen3_1.7b --num_gpus 2 --gpus 0,1
#   bash run_experiments.sh sweep --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1
#   bash run_experiments.sh evolve --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1 --rounds 3
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_SCRIPTS="$SCRIPT_DIR/experiments/scripts"
EXP_CONFIGS="$SCRIPT_DIR/experiments/configs"

# ---- Helper: expand config name to full path ----
expand_config() {
    local name="$1"
    if [[ -f "$name" ]]; then
        echo "$name"
    elif [[ -f "$EXP_CONFIGS/base/${name}.yaml" ]]; then
        echo "$EXP_CONFIGS/base/${name}.yaml"
    else
        echo "ERROR: Config not found: $name" >&2
        echo "  Tried: $name, $EXP_CONFIGS/base/${name}.yaml" >&2
        echo "  Available configs:" >&2
        ls "$EXP_CONFIGS/base/"*.yaml 2>/dev/null | sed 's/.*\//    /' | sed 's/\.yaml$//' >&2
        exit 1
    fi
}

expand_sweep() {
    local name="$1"
    if [[ -f "$name" ]]; then
        echo "$name"
    elif [[ -f "$EXP_CONFIGS/sweeps/${name}.yaml" ]]; then
        echo "$EXP_CONFIGS/sweeps/${name}.yaml"
    else
        echo "ERROR: Sweep config not found: $name" >&2
        echo "  Tried: $name, $EXP_CONFIGS/sweeps/${name}.yaml" >&2
        echo "  Available sweeps:" >&2
        ls "$EXP_CONFIGS/sweeps/"*.yaml 2>/dev/null | sed 's/.*\//    /' | sed 's/\.yaml$//' >&2
        exit 1
    fi
}

# ---- Parse subcommand ----
if [[ "$#" -lt 1 ]]; then
    echo "Usage: bash run_experiments.sh {train|sweep|evolve} [OPTIONS]"
    echo ""
    echo "Subcommands:"
    echo "  train   Run a single training experiment"
    echo "  sweep   Run a hyperparameter sweep"
    echo "  evolve  Run self-evolving RL training"
    echo ""
    echo "Config names auto-expand:"
    echo "  --config grpo_qwen3_1.7b  →  experiments/configs/base/grpo_qwen3_1.7b.yaml"
    echo "  --sweep lr_sweep          →  experiments/configs/sweeps/lr_sweep.yaml"
    echo ""
    echo "Available base configs:"
    ls "$EXP_CONFIGS/base/"*.yaml 2>/dev/null | sed 's/.*\//  /' | sed 's/\.yaml$//' || echo "  (none)"
    echo ""
    echo "Available sweep configs:"
    ls "$EXP_CONFIGS/sweeps/"*.yaml 2>/dev/null | sed 's/.*\//  /' | sed 's/\.yaml$//' || echo "  (none)"
    echo ""
    echo "Examples:"
    echo "  bash run_experiments.sh train --config grpo_qwen3_1.7b --gpus 0"
    echo "  bash run_experiments.sh sweep --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1"
    echo "  bash run_experiments.sh evolve --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1 --rounds 3"
    exit 0
fi

SUBCMD="$1"
shift

# ---- Rewrite --config and --sweep args to full paths, pass everything else through ----
ARGS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config)
            ARGS+=("$1" "$(expand_config "$2")")
            shift 2 ;;
        --sweep)
            ARGS+=("--sweep_config" "$(expand_sweep "$2")")
            shift 2 ;;
        *)
            ARGS+=("$1")
            shift ;;
    esac
done

case "$SUBCMD" in
    train)
        exec bash "$EXP_SCRIPTS/train.sh" "${ARGS[@]}"
        ;;
    sweep)
        # Rewrite --config to --base_config for sweep.sh
        REWRITTEN=()
        for arg in "${ARGS[@]}"; do
            if [[ "$arg" == "--config" ]]; then
                REWRITTEN+=("--base_config")
            else
                REWRITTEN+=("$arg")
            fi
        done
        exec bash "$EXP_SCRIPTS/sweep.sh" "${REWRITTEN[@]}"
        ;;
    evolve)
        # Rewrite --config to --base_config for self_evolve.sh
        REWRITTEN=()
        for arg in "${ARGS[@]}"; do
            if [[ "$arg" == "--config" ]]; then
                REWRITTEN+=("--base_config")
            else
                REWRITTEN+=("$arg")
            fi
        done
        exec bash "$EXP_SCRIPTS/self_evolve.sh" "${REWRITTEN[@]}"
        ;;
    *)
        echo "ERROR: Unknown subcommand: $SUBCMD" >&2
        echo "Use: train, sweep, or evolve" >&2
        exit 1 ;;
esac
