# Experiment Framework Design

**Date**: 2026-02-26
**Status**: Approved

## Overview

Restructure training scripts into a config-driven experiment framework under `experiments/`.
Supports two tasks:
1. Hyperparameter search with Qwen3-1.7B / DeepSeek-R1-Distill-Qwen-1.5B on 2x RTX 4090
2. Self-evolve RL training loop

## Hardware & Models

- **GPUs**: 2x RTX 4090 (24GB each), extensible to N=4/6
- **Models**: `Qwen/Qwen3-1.7B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Algorithms**: GRPO and PPO
- **GPU allocation**: Only user-specified GPUs via `--gpus`, no auto-discovery

## Directory Structure

```
experiments/
├── configs/
│   ├── base/                           # Full config per model x algorithm
│   │   ├── grpo_qwen3_1.7b.yaml
│   │   ├── grpo_deepseek_1.5b.yaml
│   │   ├── ppo_qwen3_1.7b.yaml
│   │   └── ppo_deepseek_1.5b.yaml
│   └── sweeps/                         # Override-only sweep definitions
│       ├── lr_sweep.yaml
│       └── kl_sweep.yaml
├── scripts/
│   ├── parse_config.py                 # YAML parser + config merger
│   ├── train.sh                        # Single experiment launcher
│   ├── sweep.sh                        # Hyperparameter sweep orchestrator
│   └── self_evolve.sh                  # Self-evolve RL loop
├── logs/                               # gitignore - local log backup
└── README.md

run_experiments.sh                      # Root-level shortcut entry point
```

## Cleanup

Delete these root-level scripts (functionality migrated to `experiments/`):
- `train_qwen3_4b_single_gpu.sh`
- `run_self_evolve_rl.sh`
- `train_grpo_math_tune_ray.sh`

## Script Call Hierarchy

```
run_experiments.sh → experiments/scripts/{train,sweep,self_evolve}.sh
self_evolve.sh → sweep.sh → train.sh → parse_config.py → verl
```

## Config Design

### Base Config (YAML)

Each base config is a complete parameter set. Example structure:

```yaml
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

### Sweep Config (override-only)

```yaml
sweep_name: "lr_sweep"
sweep_param: "actor.optim.lr"
values:
  - 5e-7
  - 1e-6
  - 2e-6
  - 5e-6
```

## parse_config.py

Python helper for YAML processing. Depends only on `pyyaml` + stdlib.

**Functions**:
- Read YAML and flatten to verl CLI arg format (`actor.optim.lr` → `actor_rollout_ref.actor.optim.lr`)
- Merge `--override` params on top of base config
- Output formats: `flat` (key=value lines), `wandb` (JSON), `header` (log metadata text)
- Parse sweep configs into override lists

## train.sh — Single Experiment Launcher

**Arguments**: `--config`, `--num_gpus`, `--gpus`, `--override`, `--suffix`, `--output_dir`

**Flow**:
1. Parse CLI args
2. Call parse_config.py to read YAML + merge overrides
3. Auto-adjust for `--num_gpus`: set `trainer.n_gpus_per_node`, adjust offload/batch settings
4. Generate run_name: `{algorithm}_{model}_{num_gpus}gpu_{lr}_{timestamp}`
5. Write metadata header to local log file (model, params, GPU count, git commit, etc.)
6. Upload config to W&B
7. Set `CUDA_VISIBLE_DEVICES` to `--gpus`, start Ray, launch verl training
8. Tee output to log file

## sweep.sh — Hyperparameter Sweep

**Arguments**: `--base_config`, `--sweep_config` (or `--sweep_param` + `--sweep_values`), `--gpus`, `--parallel`

**Flow**:
1. Parse sweep config → list of overrides
2. `--parallel N` runs N experiments concurrently, each on one of the specified GPUs
3. GPU assignment: round-robin across `--gpus` list only, never use other GPUs
4. Each experiment calls train.sh with appropriate `--gpus` (single GPU) and `--override`
5. Wait for all, print summary with log paths

## self_evolve.sh — Self-Evolve RL

**Arguments**: `--base_config`, `--sweep_config`, `--gpus`, `--rounds`, `--epochs_per_round`

**Flow**:
1. For each round: call sweep.sh with current model path
2. After round completes: extract val/correctness from logs, pick best checkpoint
3. Update model path to best checkpoint, start next round
4. Print per-round summary and final comparison

## Experiment Tracking (3 layers)

1. **Git**: `experiments/configs/` versioned in git (scripts + configs)
2. **Local logs**: `experiments/logs/{run_name}_{timestamp}.log` with metadata header (gitignored)
3. **W&B**: Full config JSON + real-time training metrics

### Log Metadata Header Format

```
=============================================
Run: grpo_qwen3-1.7b_2gpu_1e-6_20260226_143022
Date: 2026-02-26 14:30:22
Model: Qwen/Qwen3-1.7B
Algorithm: grpo
GPUs: 2 (CUDA_VISIBLE_DEVICES=0,1)
Config: experiments/configs/base/grpo_qwen3_1.7b.yaml
Overrides: actor.optim.lr=5e-7
Key params:
  lr=5e-7, kl_loss_coef=0.0001, rollout_n=4
  batch_size=32, max_prompt=1024, max_response=3072
  temperature=0.6, entropy_coeff=0.001
Git commit: abc1234
=============================================
```

## run_experiments.sh — Root Shortcut

Simplifies common commands by auto-expanding config names to full paths:

```bash
bash run_experiments.sh train --config grpo_qwen3_1.7b --num_gpus 2 --gpus 0,1
bash run_experiments.sh sweep --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1
bash run_experiments.sh evolve --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1 --rounds 3
```

## Environment

- Python venv on server (devcontainer to be added later)
- SSH to training server
- `pyyaml` as only additional dependency for parse_config.py
