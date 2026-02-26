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

# Dry run (parse config, print header, but don't start training)
bash run_experiments.sh train --config grpo_qwen3_1.7b --gpus 0 --dry_run
```

## Structure

```
experiments/
├── configs/
│   ├── base/             # Full configs: {algorithm}_{model}.yaml
│   └── sweeps/           # Override-only sweep definitions
├── scripts/
│   ├── parse_config.py   # YAML → verl CLI args / W&B JSON / log header
│   ├── train.sh          # Single experiment launcher
│   ├── sweep.sh          # Sweep orchestrator (calls train.sh)
│   └── self_evolve.sh    # Self-evolve RL loop (calls sweep.sh)
└── logs/                 # Local log backup (gitignored)
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

1. **Git** - configs and scripts versioned in the repository
2. **Local logs** - `experiments/logs/{run_name}.log` with metadata header (model, params, GPU count, git commit)
3. **W&B** - full config JSON + real-time training metrics

## GPU Management

GPUs must be explicitly specified via `--gpus`. The framework never auto-discovers or uses GPUs not in your `--gpus` list.

```bash
# Use only GPU 0
--gpus 0

# Use GPUs 0 and 1
--gpus 0,1

# Use GPUs 2,3,4,5 (e.g. on a larger machine)
--gpus 2,3,4,5
```

## Adding New Configs

**New model:** Copy an existing base config and change `model.path` and `trainer.project_name`.

**New sweep:** Create a YAML in `experiments/configs/sweeps/`:
```yaml
sweep_name: "my_sweep"
sweep_param: "actor.entropy_coeff"
values:
  - 0.0005
  - 0.001
  - 0.003
```

## Script Call Hierarchy

```
run_experiments.sh
  └── experiments/scripts/train.sh      (single experiment)
  └── experiments/scripts/sweep.sh      (calls train.sh per experiment)
  └── experiments/scripts/self_evolve.sh (calls sweep.sh per round)

All scripts call parse_config.py to read YAML configs.
```
