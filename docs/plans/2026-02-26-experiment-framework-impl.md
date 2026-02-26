# Experiment Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ad-hoc root-level training scripts with a config-driven experiment framework under `experiments/`, supporting GRPO/PPO on Qwen3-1.7B and DeepSeek-R1-Distill-Qwen-1.5B with W&B tracking.

**Architecture:** YAML base configs define full parameter sets per model/algorithm. A Python helper (`parse_config.py`) reads YAML, merges overrides, and outputs verl CLI args. Shell scripts (`train.sh` → `sweep.sh` → `self_evolve.sh`) form a layered launcher system. `run_experiments.sh` at root is a shortcut entry point.

**Tech Stack:** Bash, Python (pyyaml + stdlib), verl framework, W&B, Ray

**Design doc:** `docs/plans/2026-02-26-experiment-framework-design.md`

---

### Task 1: Create directory structure and gitignore

**Files:**
- Create: `experiments/configs/base/` (directory)
- Create: `experiments/configs/sweeps/` (directory)
- Create: `experiments/scripts/` (directory)
- Create: `experiments/logs/.gitkeep`
- Modify: `.gitignore`

**Step 1: Create the directory tree**

```bash
mkdir -p experiments/configs/base
mkdir -p experiments/configs/sweeps
mkdir -p experiments/scripts
mkdir -p experiments/logs
touch experiments/logs/.gitkeep
```

**Step 2: Add experiments/logs/ to .gitignore**

Append to `.gitignore`:

```
# Experiment logs (tracked via W&B, local backup only)
experiments/logs/*.log
```

**Step 3: Commit**

```bash
git add experiments/ .gitignore
git commit -m "scaffold: create experiments/ directory structure"
```

---

### Task 2: Write parse_config.py

**Files:**
- Create: `experiments/scripts/parse_config.py`

This is the core Python utility. It must handle:
1. Reading a YAML config file
2. Flattening nested keys to verl CLI format (with `actor_rollout_ref.` prefix mapping)
3. Merging `--override key=value` params
4. Outputting in 3 formats: `flat`, `wandb` (JSON), `header` (log metadata)
5. Parsing sweep config files into override lists

**Step 1: Write parse_config.py**

```python
#!/usr/bin/env python3
"""
Config parser for the experiment framework.

Reads YAML configs, merges overrides, outputs in formats consumable by
train.sh (flat key=value), W&B (JSON), or log files (header text).

Usage:
    # Output flat key=value pairs for verl CLI
    python parse_config.py --config base.yaml --format flat

    # With overrides
    python parse_config.py --config base.yaml --override actor.optim.lr=5e-7

    # Output JSON for W&B
    python parse_config.py --config base.yaml --format wandb

    # Output log header text
    python parse_config.py --config base.yaml --format header \
        --run_name "my_run" --gpus "0,1" --num_gpus 2

    # Parse sweep config → list of override strings
    python parse_config.py --sweep sweeps/lr_sweep.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


# Mapping from our short YAML keys to verl's CLI arg prefixes.
# Our YAML uses a clean hierarchy; verl expects actor_rollout_ref.* prefixes.
VERL_PREFIX_MAP = {
    "actor": "actor_rollout_ref.actor",
    "rollout": "actor_rollout_ref.rollout",
    "ref": "actor_rollout_ref.ref",
    "model": "actor_rollout_ref.model",
    "algorithm": "algorithm",
    "data": "data",
    "trainer": "trainer",
    "critic": "critic",
    "reward_model": "reward_model",
}


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict: {a: {b: 1}} -> {"a.b": 1}"""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def to_verl_args(flat_config: dict) -> dict:
    """Convert our flat keys to verl CLI arg keys.

    Example: actor.optim.lr -> actor_rollout_ref.actor.optim.lr
    """
    verl_args = {}
    for key, value in flat_config.items():
        top_level = key.split(".")[0]
        rest = ".".join(key.split(".")[1:]) if "." in key else ""

        if top_level in VERL_PREFIX_MAP:
            prefix = VERL_PREFIX_MAP[top_level]
            verl_key = f"{prefix}.{rest}" if rest else prefix
        else:
            verl_key = key

        verl_args[verl_key] = value
    return verl_args


def apply_overrides(flat_config: dict, overrides: list[str]) -> dict:
    """Apply key=value overrides to flat config dict.

    Override keys use our short format (e.g. actor.optim.lr),
    not the verl prefix format.
    """
    for override in overrides:
        if "=" not in override:
            print(f"WARNING: Skipping invalid override (no '='): {override}", file=sys.stderr)
            continue
        key, value = override.split("=", 1)
        # Try to parse value as number/bool
        flat_config[key] = parse_value(value)
    return flat_config


def parse_value(value: str):
    """Parse string value to appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def format_flat(verl_args: dict) -> str:
    """Output key=value lines for shell consumption."""
    lines = []
    for key, value in sorted(verl_args.items()):
        lines.append(f"{key}={value}")
    return "\n".join(lines)


def format_wandb(flat_config: dict) -> str:
    """Output JSON dict for W&B config upload."""
    # Use the original (non-verl-prefixed) keys for readability in W&B
    serializable = {}
    for k, v in flat_config.items():
        serializable[k] = v
    return json.dumps(serializable, indent=2, default=str)


def format_header(flat_config: dict, run_name: str = "", gpus: str = "", num_gpus: int = 1,
                   config_path: str = "", overrides: list[str] = None) -> str:
    """Output formatted metadata header for log files."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_path = flat_config.get("model.path", "unknown")
    algorithm = flat_config.get("algorithm.name", flat_config.get("algorithm.adv_estimator", "unknown"))
    lr = flat_config.get("actor.optim.lr", "?")
    kl_loss_coef = flat_config.get("actor.kl_loss_coef", "?")
    rollout_n = flat_config.get("rollout.n", "?")
    batch_size = flat_config.get("data.train_batch_size", "?")
    max_prompt = flat_config.get("data.max_prompt_length", "?")
    max_response = flat_config.get("data.max_response_length", "?")
    temperature = flat_config.get("rollout.temperature", "?")
    entropy_coeff = flat_config.get("actor.entropy_coeff", "?")

    override_str = ", ".join(overrides) if overrides else "none"

    # Try to get git commit
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    header = f"""=============================================
Run: {run_name}
Date: {now}
Model: {model_path}
Algorithm: {algorithm}
GPUs: {num_gpus} (CUDA_VISIBLE_DEVICES={gpus})
Config: {config_path}
Overrides: {override_str}
Key params:
  lr={lr}, kl_loss_coef={kl_loss_coef}, rollout_n={rollout_n}
  batch_size={batch_size}, max_prompt={max_prompt}, max_response={max_response}
  temperature={temperature}, entropy_coeff={entropy_coeff}
Git commit: {git_commit}
============================================="""
    return header


def parse_sweep(sweep_path: str) -> tuple[str, str, list]:
    """Parse a sweep config file.

    Returns: (sweep_name, sweep_param, list_of_values)
    """
    config = load_yaml(sweep_path)
    return config["sweep_name"], config["sweep_param"], config["values"]


def main():
    parser = argparse.ArgumentParser(description="Experiment config parser")
    parser.add_argument("--config", type=str, help="Path to base YAML config")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Override params: key=value (use our short keys, e.g. actor.optim.lr=5e-7)")
    parser.add_argument("--format", type=str, choices=["flat", "wandb", "header"], default="flat",
                        help="Output format")
    parser.add_argument("--sweep", type=str, help="Path to sweep YAML config (outputs override list)")

    # Header-specific args
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()

    # Sweep mode: just output the overrides
    if args.sweep:
        sweep_name, sweep_param, values = parse_sweep(args.sweep)
        for v in values:
            print(f"{sweep_param}={v}")
        return

    if not args.config:
        parser.error("--config is required unless --sweep is used")

    # Load and flatten config
    raw_config = load_yaml(args.config)
    flat_config = flatten_dict(raw_config)

    # Apply overrides
    if args.override:
        flat_config = apply_overrides(flat_config, args.override)

    # Output in requested format
    if args.format == "flat":
        verl_args = to_verl_args(flat_config)
        print(format_flat(verl_args))
    elif args.format == "wandb":
        print(format_wandb(flat_config))
    elif args.format == "header":
        print(format_header(
            flat_config,
            run_name=args.run_name,
            gpus=args.gpus,
            num_gpus=args.num_gpus,
            config_path=args.config,
            overrides=args.override,
        ))


if __name__ == "__main__":
    main()
```

**Step 2: Verify parse_config.py runs without errors (dry run)**

```bash
python experiments/scripts/parse_config.py --help
```

Expected: Help text printed, no import errors.

**Step 3: Commit**

```bash
git add experiments/scripts/parse_config.py
git commit -m "feat: add parse_config.py - YAML config parser for experiment framework"
```

---

### Task 3: Write base YAML configs

**Files:**
- Create: `experiments/configs/base/grpo_qwen3_1.7b.yaml`
- Create: `experiments/configs/base/grpo_deepseek_1.5b.yaml`
- Create: `experiments/configs/base/ppo_qwen3_1.7b.yaml`
- Create: `experiments/configs/base/ppo_deepseek_1.5b.yaml`

Parameters are derived from the existing `train_qwen3_4b_single_gpu.sh` (adapted for smaller models) and `verl/trainer/config/ppo_trainer.yaml`.

**Step 1: Write grpo_qwen3_1.7b.yaml**

```yaml
# Base config: GRPO training for Qwen3-1.7B on RTX 4090
# Adapted from train_qwen3_4b_single_gpu.sh with adjustments for 1.7B model size

model:
  path: "Qwen/Qwen3-1.7B"
  enable_gradient_checkpointing: true
  use_remove_padding: true

algorithm:
  name: grpo
  adv_estimator: grpo
  kl_ctrl:
    kl_coef: 0.001

data:
  train_files: "./data/simplelr_qwen_level3to5/train.parquet"
  val_files: "./data/simplelr_qwen_level3to5/test.parquet"
  train_batch_size: 32
  val_batch_size: 50
  max_prompt_length: 1024
  max_response_length: 3072

actor:
  optim:
    lr: 1e-6
  ppo_mini_batch_size: 32
  ppo_micro_batch_size_per_gpu: 1
  use_kl_loss: true
  kl_loss_coef: 0.0001
  entropy_coeff: 0.001
  clip_ratio: 0.2
  kl_loss_type: "low_var_kl"
  fsdp_config:
    param_offload: false
    grad_offload: false
    optimizer_offload: true

rollout:
  name: hf
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  n: 4
  gpu_memory_utilization: 0.4
  tensor_model_parallel_size: 1
  micro_rollout_batch_size: 4
  log_prob_micro_batch_size: 2

ref:
  log_prob_micro_batch_size: 2
  fsdp_config:
    param_offload: true

trainer:
  logger: "['console','wandb']"
  project_name: "qwen3_1.7b_grpo"
  n_gpus_per_node: 1
  nnodes: 1
  save_freq: 5
  test_freq: 5
  total_epochs: 20
  remove_previous_ckpt: false
  critic_warmup: 0
```

**Step 2: Write grpo_deepseek_1.5b.yaml**

Same structure, key differences:
- `model.path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"`
- `trainer.project_name: "deepseek_1.5b_grpo"`

```yaml
# Base config: GRPO training for DeepSeek-R1-Distill-Qwen-1.5B on RTX 4090

model:
  path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  enable_gradient_checkpointing: true
  use_remove_padding: true

algorithm:
  name: grpo
  adv_estimator: grpo
  kl_ctrl:
    kl_coef: 0.001

data:
  train_files: "./data/simplelr_qwen_level3to5/train.parquet"
  val_files: "./data/simplelr_qwen_level3to5/test.parquet"
  train_batch_size: 32
  val_batch_size: 50
  max_prompt_length: 1024
  max_response_length: 3072

actor:
  optim:
    lr: 1e-6
  ppo_mini_batch_size: 32
  ppo_micro_batch_size_per_gpu: 1
  use_kl_loss: true
  kl_loss_coef: 0.0001
  entropy_coeff: 0.001
  clip_ratio: 0.2
  kl_loss_type: "low_var_kl"
  fsdp_config:
    param_offload: false
    grad_offload: false
    optimizer_offload: true

rollout:
  name: hf
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  n: 4
  gpu_memory_utilization: 0.4
  tensor_model_parallel_size: 1
  micro_rollout_batch_size: 4
  log_prob_micro_batch_size: 2

ref:
  log_prob_micro_batch_size: 2
  fsdp_config:
    param_offload: true

trainer:
  logger: "['console','wandb']"
  project_name: "deepseek_1.5b_grpo"
  n_gpus_per_node: 1
  nnodes: 1
  save_freq: 5
  test_freq: 5
  total_epochs: 20
  remove_previous_ckpt: false
  critic_warmup: 0
```

**Step 3: Write ppo_qwen3_1.7b.yaml**

Key differences from GRPO:
- `algorithm.name: ppo`, `algorithm.adv_estimator: gae`
- `actor.use_kl_loss: false`
- Adds `critic` section
- `trainer.project_name: "qwen3_1.7b_ppo"`

```yaml
# Base config: PPO training for Qwen3-1.7B on RTX 4090
# PPO requires a critic model, which increases VRAM usage.
# May need 2+ GPUs or aggressive offloading.

model:
  path: "Qwen/Qwen3-1.7B"
  enable_gradient_checkpointing: true
  use_remove_padding: true

algorithm:
  name: ppo
  adv_estimator: gae
  gamma: 1.0
  lam: 1.0
  kl_penalty: kl
  kl_ctrl:
    type: fixed
    kl_coef: 0.001

data:
  train_files: "./data/simplelr_qwen_level3to5/train.parquet"
  val_files: "./data/simplelr_qwen_level3to5/test.parquet"
  train_batch_size: 32
  val_batch_size: 50
  max_prompt_length: 1024
  max_response_length: 3072

actor:
  optim:
    lr: 1e-6
  ppo_mini_batch_size: 32
  ppo_micro_batch_size_per_gpu: 1
  use_kl_loss: false
  kl_loss_coef: 0.001
  entropy_coeff: 0.001
  clip_ratio: 0.2
  kl_loss_type: "low_var_kl"
  fsdp_config:
    param_offload: false
    grad_offload: false
    optimizer_offload: true

rollout:
  name: hf
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  n: 1
  gpu_memory_utilization: 0.4
  tensor_model_parallel_size: 1
  micro_rollout_batch_size: 4
  log_prob_micro_batch_size: 2

ref:
  log_prob_micro_batch_size: 2
  fsdp_config:
    param_offload: true

critic:
  optim:
    lr: 1e-5
  model:
    path: "Qwen/Qwen3-1.7B"
    enable_gradient_checkpointing: true
    use_remove_padding: false
    fsdp_config:
      param_offload: true
      grad_offload: false
      optimizer_offload: true
  ppo_micro_batch_size_per_gpu: 1
  cliprange_value: 0.5
  grad_clip: 1.0

trainer:
  logger: "['console','wandb']"
  project_name: "qwen3_1.7b_ppo"
  n_gpus_per_node: 1
  nnodes: 1
  save_freq: 5
  test_freq: 5
  total_epochs: 20
  remove_previous_ckpt: false
  critic_warmup: 0
```

**Step 4: Write ppo_deepseek_1.5b.yaml**

Same as ppo_qwen3_1.7b.yaml but with:
- `model.path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"`
- `critic.model.path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"`
- `trainer.project_name: "deepseek_1.5b_ppo"`

```yaml
# Base config: PPO training for DeepSeek-R1-Distill-Qwen-1.5B on RTX 4090

model:
  path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  enable_gradient_checkpointing: true
  use_remove_padding: true

algorithm:
  name: ppo
  adv_estimator: gae
  gamma: 1.0
  lam: 1.0
  kl_penalty: kl
  kl_ctrl:
    type: fixed
    kl_coef: 0.001

data:
  train_files: "./data/simplelr_qwen_level3to5/train.parquet"
  val_files: "./data/simplelr_qwen_level3to5/test.parquet"
  train_batch_size: 32
  val_batch_size: 50
  max_prompt_length: 1024
  max_response_length: 3072

actor:
  optim:
    lr: 1e-6
  ppo_mini_batch_size: 32
  ppo_micro_batch_size_per_gpu: 1
  use_kl_loss: false
  kl_loss_coef: 0.001
  entropy_coeff: 0.001
  clip_ratio: 0.2
  kl_loss_type: "low_var_kl"
  fsdp_config:
    param_offload: false
    grad_offload: false
    optimizer_offload: true

rollout:
  name: hf
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  n: 1
  gpu_memory_utilization: 0.4
  tensor_model_parallel_size: 1
  micro_rollout_batch_size: 4
  log_prob_micro_batch_size: 2

ref:
  log_prob_micro_batch_size: 2
  fsdp_config:
    param_offload: true

critic:
  optim:
    lr: 1e-5
  model:
    path: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    enable_gradient_checkpointing: true
    use_remove_padding: false
    fsdp_config:
      param_offload: true
      grad_offload: false
      optimizer_offload: true
  ppo_micro_batch_size_per_gpu: 1
  cliprange_value: 0.5
  grad_clip: 1.0

trainer:
  logger: "['console','wandb']"
  project_name: "deepseek_1.5b_ppo"
  n_gpus_per_node: 1
  nnodes: 1
  save_freq: 5
  test_freq: 5
  total_epochs: 20
  remove_previous_ckpt: false
  critic_warmup: 0
```

**Step 5: Test parse_config.py with a real config**

```bash
python experiments/scripts/parse_config.py \
  --config experiments/configs/base/grpo_qwen3_1.7b.yaml \
  --format flat
```

Expected: flat key=value output with `actor_rollout_ref.actor.optim.lr=1e-06` etc.

```bash
python experiments/scripts/parse_config.py \
  --config experiments/configs/base/grpo_qwen3_1.7b.yaml \
  --override actor.optim.lr=5e-7 \
  --format header --run_name test_run --gpus 0,1 --num_gpus 2
```

Expected: formatted metadata header with overridden lr.

**Step 6: Commit**

```bash
git add experiments/configs/base/
git commit -m "feat: add base YAML configs for GRPO/PPO x Qwen3-1.7B/DeepSeek-1.5B"
```

---

### Task 4: Write sweep configs

**Files:**
- Create: `experiments/configs/sweeps/lr_sweep.yaml`
- Create: `experiments/configs/sweeps/kl_sweep.yaml`

**Step 1: Write lr_sweep.yaml**

```yaml
sweep_name: "lr_sweep"
sweep_param: "actor.optim.lr"
values:
  - 5e-7
  - 1e-6
  - 2e-6
  - 5e-6
```

**Step 2: Write kl_sweep.yaml**

```yaml
sweep_name: "kl_sweep"
sweep_param: "actor.kl_loss_coef"
values:
  - 0.00005
  - 0.0001
  - 0.0005
  - 0.001
```

**Step 3: Test sweep parsing**

```bash
python experiments/scripts/parse_config.py --sweep experiments/configs/sweeps/lr_sweep.yaml
```

Expected output:
```
actor.optim.lr=5e-07
actor.optim.lr=1e-06
actor.optim.lr=2e-06
actor.optim.lr=5e-06
```

**Step 4: Commit**

```bash
git add experiments/configs/sweeps/
git commit -m "feat: add lr and kl sweep configs"
```

---

### Task 5: Write train.sh

**Files:**
- Create: `experiments/scripts/train.sh`

This is the core single-experiment launcher. It calls parse_config.py, sets up Ray, and launches verl.

**Step 1: Write train.sh**

```bash
#!/bin/bash
# =============================================================================
# train.sh - Universal single-experiment launcher
#
# Reads a YAML config, optionally applies overrides, starts Ray, and launches
# verl training. Writes metadata to local log and W&B.
#
# Usage:
#   bash experiments/scripts/train.sh --config experiments/configs/base/grpo_qwen3_1.7b.yaml
#   bash experiments/scripts/train.sh --config experiments/configs/base/grpo_qwen3_1.7b.yaml \
#       --num_gpus 2 --gpus 0,1
#   bash experiments/scripts/train.sh --config experiments/configs/base/grpo_qwen3_1.7b.yaml \
#       --override actor.optim.lr=5e-7 --suffix my_exp
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Defaults ----
CONFIG=""
NUM_GPUS=1
GPUS="0"
OUTPUT_DIR="./checkpoints"
SUFFIX=""
OVERRIDES=()
MODEL_PATH_OVERRIDE=""

# ---- Parse CLI arguments ----
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        --override) OVERRIDES+=("$2"); shift 2 ;;
        --model_path) MODEL_PATH_OVERRIDE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "ERROR: --config is required"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# ---- Build override args for parse_config.py ----
OVERRIDE_ARGS=()
for ov in "${OVERRIDES[@]+"${OVERRIDES[@]}"}"; do
    OVERRIDE_ARGS+=(--override "$ov")
done

# Override num_gpus in trainer config
OVERRIDE_ARGS+=(--override "trainer.n_gpus_per_node=$NUM_GPUS")

# If model path override provided (e.g., from self_evolve.sh)
if [ -n "$MODEL_PATH_OVERRIDE" ]; then
    OVERRIDE_ARGS+=(--override "model.path=$MODEL_PATH_OVERRIDE")
fi

# Multi-GPU adjustments: relax offloading when we have more VRAM
if [ "$NUM_GPUS" -gt 1 ]; then
    OVERRIDE_ARGS+=(--override "actor.fsdp_config.optimizer_offload=false")
fi

# ---- Parse config ----
PARSE_CMD="python $SCRIPT_DIR/parse_config.py --config $CONFIG ${OVERRIDE_ARGS[*]+"${OVERRIDE_ARGS[*]}"}"

# Get flat key=value pairs for verl
FLAT_ARGS=$($PARSE_CMD --format flat)

# Get metadata for log header
# Extract model name and algorithm from config for run_name
MODEL_PATH=$(echo "$FLAT_ARGS" | grep "^actor_rollout_ref.model.path=" | cut -d= -f2-)
ALGORITHM=$(echo "$FLAT_ARGS" | grep "^algorithm.adv_estimator=" | cut -d= -f2-)
LR=$(echo "$FLAT_ARGS" | grep "^actor_rollout_ref.actor.optim.lr=" | cut -d= -f2-)
MODEL_SHORT=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

RUN_NAME="${ALGORITHM}_${MODEL_SHORT}_${NUM_GPUS}gpu_lr${LR}_${TIMESTAMP}"
if [ -n "$SUFFIX" ]; then
    RUN_NAME="${RUN_NAME}_${SUFFIX}"
fi

CKPT_DIR="${OUTPUT_DIR}/${RUN_NAME}"
LOG_DIR="$PROJECT_ROOT/experiments/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# ---- Write log header ----
HEADER=$($PARSE_CMD --format header --run_name "$RUN_NAME" --gpus "$GPUS" --num_gpus "$NUM_GPUS")
echo "$HEADER" > "$LOG_FILE"
echo "" >> "$LOG_FILE"

# ---- Write W&B config JSON (for reference) ----
WANDB_JSON=$($PARSE_CMD --format wandb)

# ---- Print summary ----
echo "$HEADER"
echo ""

# ---- Environment ----
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES="$GPUS"

# ---- Launch Ray ----
ray stop --force 2>/dev/null || true
ray start --head --num-gpus "$NUM_GPUS" --num-cpus 8
sleep 3

# ---- Build verl command from flat args ----
# Read flat args into an array for the verl command
VERL_ARGS=()
while IFS= read -r line; do
    [ -n "$line" ] && VERL_ARGS+=("$line")
done <<< "$FLAT_ARGS"

# Override checkpoint dir and experiment name
VERL_ARGS+=("trainer.default_local_dir=$CKPT_DIR")
VERL_ARGS+=("trainer.experiment_name=$RUN_NAME")

# ---- Run training ----
python -m verl.trainer.main_ppo \
    "${VERL_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "============================================="
echo "Training complete: $RUN_NAME"
echo "Log: $LOG_FILE"
echo "Checkpoints: $CKPT_DIR"
echo "============================================="
```

**Step 2: Make executable and test --help path**

```bash
chmod +x experiments/scripts/train.sh
```

**Step 3: Commit**

```bash
git add experiments/scripts/train.sh
git commit -m "feat: add train.sh - universal single-experiment launcher"
```

---

### Task 6: Write sweep.sh

**Files:**
- Create: `experiments/scripts/sweep.sh`

**Step 1: Write sweep.sh**

```bash
#!/bin/bash
# =============================================================================
# sweep.sh - Hyperparameter sweep orchestrator
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

# ---- Parse CLI arguments ----
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --base_config) BASE_CONFIG="$2"; shift 2 ;;
        --sweep_config) SWEEP_CONFIG="$2"; shift 2 ;;
        --sweep_param) SWEEP_PARAM="$2"; shift 2 ;;
        --sweep_values) SWEEP_VALUES="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --override) EXTRA_OVERRIDES+=("$2"); shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$BASE_CONFIG" ]; then
    echo "ERROR: --base_config is required"
    exit 1
fi

# ---- Build override list ----
OVERRIDE_LIST=()

if [ -n "$SWEEP_CONFIG" ]; then
    # Parse sweep config via parse_config.py
    SWEEP_NAME=$(python "$SCRIPT_DIR/parse_config.py" --sweep "$SWEEP_CONFIG" | head -1 | cut -d= -f1)
    while IFS= read -r line; do
        [ -n "$line" ] && OVERRIDE_LIST+=("$line")
    done < <(python "$SCRIPT_DIR/parse_config.py" --sweep "$SWEEP_CONFIG")
elif [ -n "$SWEEP_PARAM" ] && [ -n "$SWEEP_VALUES" ]; then
    SWEEP_NAME="$SWEEP_PARAM"
    IFS=',' read -ra VALUES <<< "$SWEEP_VALUES"
    for v in "${VALUES[@]}"; do
        OVERRIDE_LIST+=("${SWEEP_PARAM}=${v}")
    done
else
    echo "ERROR: Either --sweep_config or (--sweep_param + --sweep_values) is required"
    exit 1
fi

NUM_EXPERIMENTS=${#OVERRIDE_LIST[@]}

# ---- Parse GPU list ----
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_AVAILABLE_GPUS=${#GPU_ARRAY[@]}

echo "============================================="
echo "Sweep: $SWEEP_NAME ($NUM_EXPERIMENTS experiments)"
echo "GPUs: $GPUS ($NUM_AVAILABLE_GPUS available)"
echo "Base config: $BASE_CONFIG"
echo "============================================="

# ---- Run experiments ----
# Queue experiments, max $NUM_AVAILABLE_GPUS running concurrently.
# Each experiment gets exactly 1 GPU from the specified list.
PIDS=()
GPU_PIDS=()  # tracks which PID is using which GPU slot
LOG_FILES=()
EXP_NAMES=()

for i in "${!GPU_ARRAY[@]}"; do
    GPU_PIDS[$i]=""
done

exp_idx=0
while [ "$exp_idx" -lt "$NUM_EXPERIMENTS" ]; do
    # Find a free GPU slot
    gpu_slot=-1
    while [ "$gpu_slot" -eq -1 ]; do
        for i in "${!GPU_ARRAY[@]}"; do
            pid="${GPU_PIDS[$i]}"
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                gpu_slot=$i
                break
            fi
        done
        if [ "$gpu_slot" -eq -1 ]; then
            sleep 5
        fi
    done

    # Launch experiment on this GPU
    override="${OVERRIDE_LIST[$exp_idx]}"
    gpu_id="${GPU_ARRAY[$gpu_slot]}"
    value=$(echo "$override" | cut -d= -f2)
    suffix="${SWEEP_NAME}_${value}"

    # Build override args
    TRAIN_OVERRIDE_ARGS=(--override "$override")
    for ov in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
        TRAIN_OVERRIDE_ARGS+=(--override "$ov")
    done

    echo "[Exp $((exp_idx+1))/$NUM_EXPERIMENTS] GPU $gpu_id: $override"

    bash "$SCRIPT_DIR/train.sh" \
        --config "$BASE_CONFIG" \
        --num_gpus 1 \
        --gpus "$gpu_id" \
        --output_dir "$OUTPUT_DIR" \
        --suffix "$suffix" \
        "${TRAIN_OVERRIDE_ARGS[@]}" &

    pid=$!
    PIDS+=($pid)
    GPU_PIDS[$gpu_slot]=$pid
    EXP_NAMES+=("$override")

    exp_idx=$((exp_idx + 1))
    # Brief stagger to avoid resource contention during Ray init
    sleep 3
done

# ---- Wait for all experiments ----
echo ""
echo "Waiting for all $NUM_EXPERIMENTS experiments to complete..."
failed=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "FAILED: ${EXP_NAMES[$i]}"
        ((failed++))
    else
        echo "DONE: ${EXP_NAMES[$i]}"
    fi
done

# ---- Print summary ----
echo ""
echo "============================================="
echo "Sweep complete: $SWEEP_NAME"
echo "Total: $NUM_EXPERIMENTS, Failed: $failed"
echo "Logs: experiments/logs/"
echo "============================================="

exit $failed
```

**Step 2: Make executable**

```bash
chmod +x experiments/scripts/sweep.sh
```

**Step 3: Commit**

```bash
git add experiments/scripts/sweep.sh
git commit -m "feat: add sweep.sh - hyperparameter sweep orchestrator"
```

---

### Task 7: Write self_evolve.sh

**Files:**
- Create: `experiments/scripts/self_evolve.sh`

**Step 1: Write self_evolve.sh**

```bash
#!/bin/bash
# =============================================================================
# self_evolve.sh - Self-Evolving RL training loop
#
# Runs multiple rounds of hyperparameter sweeps. After each round, selects
# the best checkpoint and uses it as the starting model for the next round.
#
# Usage:
#   bash experiments/scripts/self_evolve.sh \
#       --base_config experiments/configs/base/grpo_qwen3_1.7b.yaml \
#       --sweep_config experiments/configs/sweeps/lr_sweep.yaml \
#       --gpus 0,1 \
#       --rounds 3 \
#       --epochs_per_round 10
# =============================================================================

set -euo pipefail

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
        --base_config) BASE_CONFIG="$2"; shift 2 ;;
        --sweep_config) SWEEP_CONFIG="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --epochs_per_round) EPOCHS_PER_ROUND="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$BASE_CONFIG" ] || [ -z "$SWEEP_CONFIG" ]; then
    echo "ERROR: --base_config and --sweep_config are required"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ---- Helper: Extract best validation score from a log file ----
get_best_score() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        grep -oP "val/correctness['\"]?\s*[:=]\s*\K[0-9.]+" "$log_file" 2>/dev/null | sort -rn | head -1
    fi
    echo "0"
}

# ---- Get initial model path from config ----
CURRENT_MODEL=$(python "$SCRIPT_DIR/parse_config.py" \
    --config "$BASE_CONFIG" --format flat \
    | grep "^actor_rollout_ref.model.path=" | cut -d= -f2-)

echo "============================================="
echo "Self-Evolving RL Training"
echo "============================================="
echo "Base config:       $BASE_CONFIG"
echo "Sweep config:      $SWEEP_CONFIG"
echo "Starting model:    $CURRENT_MODEL"
echo "GPUs:              $GPUS"
echo "Rounds:            $ROUNDS"
echo "Epochs per round:  $EPOCHS_PER_ROUND"
echo "Output:            $OUTPUT_DIR"
echo "============================================="

ROUND_RESULTS=()

for round in $(seq 1 "$ROUNDS"); do
    echo ""
    echo "===== ROUND $round / $ROUNDS ====="
    echo "Model: $CURRENT_MODEL"

    ROUND_DIR="${OUTPUT_DIR}/round${round}"

    # Run sweep for this round
    bash "$SCRIPT_DIR/sweep.sh" \
        --base_config "$BASE_CONFIG" \
        --sweep_config "$SWEEP_CONFIG" \
        --gpus "$GPUS" \
        --output_dir "$ROUND_DIR" \
        --override "trainer.total_epochs=$EPOCHS_PER_ROUND" \
        --override "model.path=$CURRENT_MODEL" \
        || echo "WARNING: Some experiments in round $round failed"

    # ---- Select best checkpoint ----
    echo ""
    echo "--- Selecting best checkpoint from round $round ---"
    best_score=0
    best_ckpt=""
    best_name=""

    LOG_DIR="$PROJECT_ROOT/experiments/logs"
    for log_file in "$LOG_DIR"/*round${round}*.log 2>/dev/null; do
        [ -f "$log_file" ] || continue
        score=$(get_best_score "$log_file")
        score=${score:-0}
        name=$(basename "$log_file" .log)
        echo "  $name: score=$score"

        is_better=$(awk "BEGIN {print ($score > $best_score) ? 1 : 0}")
        if [ "$is_better" -eq 1 ]; then
            best_score=$score
            best_name=$name
            # Find checkpoint directory
            for ckpt_dir in "$ROUND_DIR"/*; do
                [ -d "$ckpt_dir" ] || continue
                latest_ckpt=$(ls -td "${ckpt_dir}"/global_step_* 2>/dev/null | head -1)
                if [ -n "$latest_ckpt" ]; then
                    best_ckpt="$latest_ckpt"
                fi
            done
        fi
    done

    if [ -n "$best_ckpt" ] && [ -d "$best_ckpt" ]; then
        echo ""
        echo "Best: $best_name (score: $best_score)"
        echo "Checkpoint: $best_ckpt"
        CURRENT_MODEL="$best_ckpt"
        ROUND_RESULTS+=("Round $round: $best_name (score=$best_score)")
    else
        echo "WARNING: No valid checkpoint found for round $round."
        echo "Continuing with current model: $CURRENT_MODEL"
        ROUND_RESULTS+=("Round $round: no improvement found")
    fi

    echo "===== END ROUND $round ====="
done

# ---- Final summary ----
echo ""
echo "============================================="
echo "Self-Evolving RL Complete"
echo "============================================="
for result in "${ROUND_RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "Final model: $CURRENT_MODEL"
echo "============================================="
```

**Step 2: Make executable**

```bash
chmod +x experiments/scripts/self_evolve.sh
```

**Step 3: Commit**

```bash
git add experiments/scripts/self_evolve.sh
git commit -m "feat: add self_evolve.sh - multi-round self-evolving RL loop"
```

---

### Task 8: Write run_experiments.sh (root shortcut)

**Files:**
- Create: `run_experiments.sh` (at project root)

**Step 1: Write run_experiments.sh**

```bash
#!/bin/bash
# =============================================================================
# run_experiments.sh - Shortcut entry point for the experiment framework
#
# Wraps experiments/scripts/{train,sweep,self_evolve}.sh with shorter syntax.
# Auto-expands config names to full paths.
#
# Usage:
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
    if [ -f "$name" ]; then
        echo "$name"
    elif [ -f "$EXP_CONFIGS/base/${name}.yaml" ]; then
        echo "$EXP_CONFIGS/base/${name}.yaml"
    else
        echo "ERROR: Config not found: $name (tried $EXP_CONFIGS/base/${name}.yaml)" >&2
        exit 1
    fi
}

expand_sweep() {
    local name="$1"
    if [ -f "$name" ]; then
        echo "$name"
    elif [ -f "$EXP_CONFIGS/sweeps/${name}.yaml" ]; then
        echo "$EXP_CONFIGS/sweeps/${name}.yaml"
    else
        echo "ERROR: Sweep config not found: $name (tried $EXP_CONFIGS/sweeps/${name}.yaml)" >&2
        exit 1
    fi
}

# ---- Parse subcommand ----
if [ "$#" -lt 1 ]; then
    echo "Usage: bash run_experiments.sh {train|sweep|evolve} [OPTIONS]"
    echo ""
    echo "Subcommands:"
    echo "  train   - Run a single training experiment"
    echo "  sweep   - Run a hyperparameter sweep"
    echo "  evolve  - Run self-evolving RL training"
    echo ""
    echo "Config names auto-expand: 'grpo_qwen3_1.7b' -> experiments/configs/base/grpo_qwen3_1.7b.yaml"
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
            ARGS+=(--config "$(expand_config "$2")")
            shift 2
            ;;
        --sweep)
            ARGS+=(--sweep_config "$(expand_sweep "$2")")
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

case "$SUBCMD" in
    train)
        exec bash "$EXP_SCRIPTS/train.sh" "${ARGS[@]}"
        ;;
    sweep)
        # Rewrite --config to --base_config for sweep.sh
        SWEEP_ARGS=()
        for i in "${!ARGS[@]}"; do
            if [ "${ARGS[$i]}" = "--config" ]; then
                SWEEP_ARGS+=("--base_config")
            else
                SWEEP_ARGS+=("${ARGS[$i]}")
            fi
        done
        exec bash "$EXP_SCRIPTS/sweep.sh" "${SWEEP_ARGS[@]}"
        ;;
    evolve)
        # Rewrite --config to --base_config for self_evolve.sh
        EVOLVE_ARGS=()
        for i in "${!ARGS[@]}"; do
            if [ "${ARGS[$i]}" = "--config" ]; then
                EVOLVE_ARGS+=("--base_config")
            else
                EVOLVE_ARGS+=("${ARGS[$i]}")
            fi
        done
        exec bash "$EXP_SCRIPTS/self_evolve.sh" "${EVOLVE_ARGS[@]}"
        ;;
    *)
        echo "Unknown subcommand: $SUBCMD"
        echo "Use: train, sweep, or evolve"
        exit 1
        ;;
esac
```

**Step 2: Make executable**

```bash
chmod +x run_experiments.sh
```

**Step 3: Commit**

```bash
git add run_experiments.sh
git commit -m "feat: add run_experiments.sh - root-level shortcut entry point"
```

---

### Task 9: Delete old scripts and clean up

**Files:**
- Delete: `train_qwen3_4b_single_gpu.sh`
- Delete: `run_self_evolve_rl.sh`
- Delete: `train_grpo_math_tune_ray.sh`

**Step 1: Remove old scripts**

```bash
git rm train_qwen3_4b_single_gpu.sh
git rm run_self_evolve_rl.sh
git rm train_grpo_math_tune_ray.sh
```

**Step 2: Commit**

```bash
git commit -m "cleanup: remove old training scripts (migrated to experiments/)"
```

---

### Task 10: Write experiments/README.md

**Files:**
- Create: `experiments/README.md`

**Step 1: Write README**

```markdown
# Experiment Framework

Config-driven training framework for RL experiments on RTX 4090 GPUs.

## Quick Start

```bash
# Single training run (1 GPU)
bash run_experiments.sh train --config grpo_qwen3_1.7b --gpus 0

# Two GPUs with FSDP
bash run_experiments.sh train --config grpo_qwen3_1.7b --num_gpus 2 --gpus 0,1

# Hyperparameter sweep (2 experiments in parallel on 2 GPUs)
bash run_experiments.sh sweep --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1

# Self-evolving RL (3 rounds)
bash run_experiments.sh evolve --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1 --rounds 3

# Override any parameter
bash run_experiments.sh train --config grpo_qwen3_1.7b --gpus 0 --override actor.optim.lr=5e-7
```

## Structure

```
experiments/
├── configs/
│   ├── base/          # Full configs: {algorithm}_{model}.yaml
│   └── sweeps/        # Override-only sweep definitions
├── scripts/
│   ├── parse_config.py   # YAML → verl CLI args
│   ├── train.sh          # Single experiment launcher
│   ├── sweep.sh          # Sweep orchestrator
│   └── self_evolve.sh    # Self-evolve RL loop
└── logs/              # Local log backup (gitignored)
```

## Available Configs

| Config | Model | Algorithm |
|--------|-------|-----------|
| `grpo_qwen3_1.7b` | Qwen/Qwen3-1.7B | GRPO |
| `grpo_deepseek_1.5b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | GRPO |
| `ppo_qwen3_1.7b` | Qwen/Qwen3-1.7B | PPO |
| `ppo_deepseek_1.5b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | PPO |

## Available Sweeps

| Sweep | Parameter | Values |
|-------|-----------|--------|
| `lr_sweep` | actor.optim.lr | 5e-7, 1e-6, 2e-6, 5e-6 |
| `kl_sweep` | actor.kl_loss_coef | 5e-5, 1e-4, 5e-4, 1e-3 |

## Experiment Tracking

Each experiment is tracked at 3 levels:
1. **Git** - configs and scripts versioned
2. **Local logs** - `experiments/logs/{run_name}.log` with metadata header
3. **W&B** - full config + real-time metrics

## GPU Management

GPUs must be explicitly specified via `--gpus`. The framework will never
auto-discover or use GPUs not in your `--gpus` list.
```

**Step 2: Commit**

```bash
git add experiments/README.md
git commit -m "docs: add experiments README with usage guide"
```
