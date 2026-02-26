.. _experiment-framework-page:

=============================
Experiment Framework
=============================

Overview
--------

The experiment framework provides a config-driven system for running RL training
experiments. It replaces the ad-hoc shell scripts that were previously in the
project root with a structured approach:

- **YAML configs** define complete experiment parameters
- **Shell scripts** provide layered launchers (single run → sweep → self-evolve)
- **W&B + local logs** track every experiment with full metadata

Hardware & Models
-----------------

The framework is designed for:

- **GPUs**: 2x RTX 4090 (24GB each), extensible to N=4/6
- **Models**: ``Qwen/Qwen3-1.7B``, ``deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B``
- **Algorithms**: GRPO and PPO

Directory Structure
-------------------

.. code-block:: text

   experiments/
   ├── configs/
   │   ├── base/                # Full configs per model x algorithm
   │   │   ├── grpo_qwen3_1.7b.yaml
   │   │   ├── grpo_deepseek_1.5b.yaml
   │   │   ├── ppo_qwen3_1.7b.yaml
   │   │   └── ppo_deepseek_1.5b.yaml
   │   └── sweeps/              # Override-only sweep definitions
   │       ├── lr_sweep.yaml
   │       └── kl_sweep.yaml
   ├── scripts/
   │   ├── parse_config.py      # YAML → verl CLI args / W&B JSON / log header
   │   ├── train.sh             # Single experiment launcher
   │   ├── sweep.sh             # Sweep orchestrator (calls train.sh)
   │   └── self_evolve.sh       # Self-evolve RL loop (calls sweep.sh)
   ├── logs/                    # Local log backup (gitignored)
   └── README.md

   run_experiments.sh            # Root-level shortcut entry point

Script Call Hierarchy
---------------------

.. code-block:: text

   run_experiments.sh
     ├── train.sh          (single experiment)
     ├── sweep.sh          (calls train.sh per experiment)
     └── self_evolve.sh    (calls sweep.sh per round)

   All scripts call parse_config.py to read YAML configs.

Quick Start
-----------

All commands use ``run_experiments.sh`` at the project root, which auto-expands
config names to full paths.

**Single training run (1 GPU):**

.. code-block:: bash

   bash run_experiments.sh train --config grpo_qwen3_1.7b --gpus 0

**Two GPUs with FSDP:**

.. code-block:: bash

   bash run_experiments.sh train --config grpo_qwen3_1.7b --num_gpus 2 --gpus 0,1

**Hyperparameter sweep (2 experiments in parallel on 2 GPUs):**

.. code-block:: bash

   bash run_experiments.sh sweep --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1

**Self-evolving RL (3 rounds):**

.. code-block:: bash

   bash run_experiments.sh evolve --config grpo_qwen3_1.7b --sweep lr_sweep --gpus 0,1 --rounds 3

**Override any parameter:**

.. code-block:: bash

   bash run_experiments.sh train --config grpo_qwen3_1.7b --gpus 0 --override actor.optim.lr=5e-7

**Dry run (test config without GPU):**

.. code-block:: bash

   bash run_experiments.sh train --config grpo_qwen3_1.7b --gpus 0 --dry_run


train.sh — Single Experiment
-----------------------------

The core launcher. Reads a YAML config, starts Ray, and launches verl training.

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--config PATH``
     - (required)
     - Path to YAML config file
   * - ``--num_gpus N``
     - 1
     - Number of GPUs for this experiment
   * - ``--gpus IDS``
     - "0"
     - Comma-separated GPU IDs (CUDA_VISIBLE_DEVICES)
   * - ``--output_dir PATH``
     - ./checkpoints
     - Checkpoint output directory
   * - ``--suffix STRING``
     -
     - Optional suffix appended to run name
   * - ``--override KEY=VALUE``
     -
     - Override config params (repeatable, uses short keys)
   * - ``--model_path PATH``
     -
     - Override model path (used by self_evolve.sh)
   * - ``--dry_run``
     -
     - Parse config and print, but skip training

**What it does:**

1. Calls ``parse_config.py`` to read YAML + merge overrides
2. Auto-adjusts for multi-GPU: sets ``n_gpus_per_node``, disables optimizer offload
3. Generates run name: ``{algorithm}_{model}_{num_gpus}gpu_lr{lr}_{timestamp}``
4. Writes metadata header to local log file
5. Saves W&B config JSON
6. Sets ``CUDA_VISIBLE_DEVICES``, starts Ray, launches verl training


sweep.sh — Hyperparameter Sweep
---------------------------------

Orchestrates multiple experiments with different parameter values. One experiment
per GPU from the specified list.

**Arguments:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--base_config PATH``
     - (required)
     - Path to base YAML config
   * - ``--sweep_config PATH``
     -
     - Path to sweep YAML config
   * - ``--sweep_param KEY``
     -
     - Parameter to sweep (inline mode)
   * - ``--sweep_values VALUES``
     -
     - Comma-separated values (inline mode)
   * - ``--gpus IDS``
     - "0"
     - Comma-separated GPU IDs (only these GPUs are used)
   * - ``--output_dir PATH``
     - ./checkpoints
     - Checkpoint output directory
   * - ``--override KEY=VALUE``
     -
     - Extra overrides for every experiment (repeatable)

**GPU queue behavior:**

- Max concurrency = number of GPUs in ``--gpus``
- Each experiment gets exactly 1 GPU
- When a GPU finishes, the next experiment takes its slot
- Only GPUs from ``--gpus`` are used — no auto-discovery
- 3-second stagger between launches to avoid Ray init contention

**Inline sweep (without config file):**

.. code-block:: bash

   bash experiments/scripts/sweep.sh \
       --base_config experiments/configs/base/grpo_qwen3_1.7b.yaml \
       --sweep_param actor.optim.lr \
       --sweep_values "5e-7,1e-6,2e-6" \
       --gpus 0,1


Experiment Tracking
-------------------

Each experiment is recorded at 3 levels:

1. **Git** — ``experiments/configs/`` versioned in the repository. Config changes
   are visible in git history.

2. **Local logs** — ``experiments/logs/{run_name}.log`` with a metadata header at
   the top of each file:

   .. code-block:: text

      =============================================
      Run: grpo_qwen3-1.7b_2gpu_lr1e-06_20260226_143022
      Date: 2026-02-26 14:30:22
      Model: Qwen/Qwen3-1.7B
      Algorithm: grpo
      GPUs: 2 (CUDA_VISIBLE_DEVICES=0,1)
      Config: experiments/configs/base/grpo_qwen3_1.7b.yaml
      Overrides: actor.optim.lr=5e-7
      Key params:
        lr=5e-07, kl_loss_coef=0.0001, rollout_n=4
        batch_size=32, max_prompt=1024, max_response=3072
        temperature=0.6, entropy_coeff=0.001
      Git commit: abc1234
      =============================================

3. **W&B** — Full config JSON uploaded + real-time training metrics. Set
   ``WANDB_API_KEY`` environment variable to enable.


GPU Management
--------------

GPUs must be explicitly specified via ``--gpus``. The framework never
auto-discovers or uses GPUs outside your list.

.. code-block:: bash

   --gpus 0            # Use only GPU 0
   --gpus 0,1          # Use GPUs 0 and 1
   --gpus 2,3,4,5      # Use GPUs 2-5 on a larger machine


Adding New Configurations
-------------------------

**New model:**

Copy an existing base config and change:

- ``model.path`` — the HuggingFace model name or local path
- ``trainer.project_name`` — W&B project name
- For PPO: also update ``critic.model.path``

**New sweep:**

Create a YAML file in ``experiments/configs/sweeps/``:

.. code-block:: yaml

   sweep_name: "entropy_sweep"
   sweep_param: "actor.entropy_coeff"
   values:
     - 0.0005
     - 0.001
     - 0.003

Then use it:

.. code-block:: bash

   bash run_experiments.sh sweep --config grpo_qwen3_1.7b --sweep entropy_sweep --gpus 0,1
