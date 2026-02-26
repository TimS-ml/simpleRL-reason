.. _config-reference-page:

=============================
Config Reference
=============================

This page documents the YAML config format used by the experiment framework.

Config Types
------------

**Base configs** (``experiments/configs/base/``) define a complete set of
parameters for a model + algorithm combination. Each file is self-contained.

**Sweep configs** (``experiments/configs/sweeps/``) define override-only
parameter variations for hyperparameter search.

Available Base Configs
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 15 20

   * - Config Name
     - Model
     - Algorithm
     - W&B Project
   * - ``grpo_qwen3_1.7b``
     - Qwen/Qwen3-1.7B
     - GRPO
     - qwen3_1.7b_grpo
   * - ``grpo_deepseek_1.5b``
     - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
     - GRPO
     - deepseek_1.5b_grpo
   * - ``ppo_qwen3_1.7b``
     - Qwen/Qwen3-1.7B
     - PPO
     - qwen3_1.7b_ppo
   * - ``ppo_deepseek_1.5b``
     - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
     - PPO
     - deepseek_1.5b_ppo

Available Sweep Configs
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Sweep Name
     - Parameter
     - Values
   * - ``lr_sweep``
     - actor.optim.lr
     - 5e-7, 1e-6, 2e-6, 5e-6
   * - ``kl_sweep``
     - actor.kl_loss_coef
     - 5e-5, 1e-4, 5e-4, 1e-3


Base Config Format
------------------

A base config has the following top-level sections. Each maps to verl's CLI
argument namespace.

model
^^^^^

.. code-block:: yaml

   model:
     path: "Qwen/Qwen3-1.7B"           # HuggingFace model name or local path
     enable_gradient_checkpointing: true # Save VRAM at cost of compute
     use_remove_padding: true            # Remove padding for efficiency

Maps to: ``actor_rollout_ref.model.*``

algorithm
^^^^^^^^^

.. code-block:: yaml

   # For GRPO:
   algorithm:
     name: grpo
     adv_estimator: grpo
     kl_ctrl:
       kl_coef: 0.001

   # For PPO:
   algorithm:
     name: ppo
     adv_estimator: gae
     gamma: 1.0
     lam: 1.0
     kl_penalty: kl
     kl_ctrl:
       type: fixed
       kl_coef: 0.001

Maps to: ``algorithm.*``

data
^^^^

.. code-block:: yaml

   data:
     train_files: "./data/simplelr_qwen_level3to5/train.parquet"
     val_files: "./data/simplelr_qwen_level3to5/test.parquet"
     train_batch_size: 32       # Total batch size across all GPUs
     val_batch_size: 50
     max_prompt_length: 1024
     max_response_length: 3072

Maps to: ``data.*``

actor
^^^^^

.. code-block:: yaml

   actor:
     optim:
       lr: 1e-6                           # Learning rate
     ppo_mini_batch_size: 32              # Mini-batch for PPO update
     ppo_micro_batch_size_per_gpu: 1      # Micro-batch per GPU (controls VRAM)
     use_kl_loss: true                    # true for GRPO, false for PPO
     kl_loss_coef: 0.0001                 # KL divergence loss coefficient
     entropy_coeff: 0.001                 # Entropy bonus coefficient
     clip_ratio: 0.2                      # PPO clip ratio
     kl_loss_type: "low_var_kl"           # KL loss variant
     fsdp_config:
       param_offload: false               # Offload params to CPU
       grad_offload: false                # Offload gradients to CPU
       optimizer_offload: true            # Offload optimizer to CPU (single GPU)

Maps to: ``actor_rollout_ref.actor.*``

**Note:** When ``--num_gpus > 1``, ``train.sh`` automatically disables
``optimizer_offload`` since multi-GPU has enough VRAM.

rollout
^^^^^^^

.. code-block:: yaml

   rollout:
     name: hf                             # hf (HuggingFace) or vllm
     temperature: 0.6                     # Sampling temperature
     top_p: 0.95                          # Nucleus sampling
     top_k: 20                            # Top-k sampling
     n: 4                                 # Rollouts per prompt (GRPO uses >1)
     gpu_memory_utilization: 0.4          # Fraction of GPU memory for rollout
     tensor_model_parallel_size: 1        # TP size for rollout model
     micro_rollout_batch_size: 4          # Micro-batch for rollout
     log_prob_micro_batch_size: 2         # Micro-batch for log-prob computation

Maps to: ``actor_rollout_ref.rollout.*``

ref
^^^

.. code-block:: yaml

   ref:
     log_prob_micro_batch_size: 2
     fsdp_config:
       param_offload: true                # Reference model offloaded to save VRAM

Maps to: ``actor_rollout_ref.ref.*``

critic (PPO only)
^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   critic:
     optim:
       lr: 1e-5                           # Critic learning rate (typically higher)
     model:
       path: "Qwen/Qwen3-1.7B"           # Usually same as actor model
       enable_gradient_checkpointing: true
       use_remove_padding: false
       fsdp_config:
         param_offload: true              # Offload critic to save VRAM
         grad_offload: false
         optimizer_offload: true
     ppo_micro_batch_size_per_gpu: 1
     cliprange_value: 0.5
     grad_clip: 1.0

Maps to: ``critic.*``

**Not present in GRPO configs** — GRPO does not use a learned critic.

trainer
^^^^^^^

.. code-block:: yaml

   trainer:
     logger: "['console','wandb']"        # Logging backends
     project_name: "qwen3_1.7b_grpo"      # W&B project name
     n_gpus_per_node: 1                   # Auto-set by train.sh --num_gpus
     nnodes: 1                            # Number of nodes (always 1 for us)
     save_freq: 5                         # Save checkpoint every N epochs
     test_freq: 5                         # Run validation every N epochs
     total_epochs: 20                     # Total training epochs
     remove_previous_ckpt: false          # Keep all checkpoints
     critic_warmup: 0                     # Critic warmup steps

Maps to: ``trainer.*``


Key Mapping: YAML → verl CLI
-----------------------------

The experiment framework uses short, clean YAML keys. ``parse_config.py``
maps them to verl's full CLI argument names:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - YAML Section
     - verl CLI Prefix
     - Example
   * - ``model.*``
     - ``actor_rollout_ref.model.*``
     - ``model.path`` → ``actor_rollout_ref.model.path``
   * - ``actor.*``
     - ``actor_rollout_ref.actor.*``
     - ``actor.optim.lr`` → ``actor_rollout_ref.actor.optim.lr``
   * - ``rollout.*``
     - ``actor_rollout_ref.rollout.*``
     - ``rollout.n`` → ``actor_rollout_ref.rollout.n``
   * - ``ref.*``
     - ``actor_rollout_ref.ref.*``
     - ``ref.fsdp_config.param_offload`` → ``actor_rollout_ref.ref.fsdp_config.param_offload``
   * - ``algorithm.*``
     - ``algorithm.*`` (passthrough)
     - ``algorithm.kl_ctrl.kl_coef`` → ``algorithm.kl_ctrl.kl_coef``
   * - ``data.*``
     - ``data.*`` (passthrough)
     - ``data.train_batch_size`` → ``data.train_batch_size``
   * - ``trainer.*``
     - ``trainer.*`` (passthrough)
     - ``trainer.total_epochs`` → ``trainer.total_epochs``
   * - ``critic.*``
     - ``critic.*`` (passthrough)
     - ``critic.optim.lr`` → ``critic.optim.lr``

**Override keys use the short format**, e.g.:

.. code-block:: bash

   --override actor.optim.lr=5e-7
   --override rollout.temperature=0.8


Sweep Config Format
-------------------

A sweep config defines one parameter to vary across experiments:

.. code-block:: yaml

   sweep_name: "lr_sweep"             # Human-readable name (used in log suffixes)
   sweep_param: "actor.optim.lr"      # Parameter key (short format)
   values:                            # List of values to try
     - 5e-7
     - 1e-6
     - 2e-6
     - 5e-6

Each value generates one experiment with the base config + that single override.


GRPO vs PPO: Key Differences
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Parameter
     - GRPO
     - PPO
   * - ``algorithm.adv_estimator``
     - ``grpo``
     - ``gae``
   * - ``actor.use_kl_loss``
     - ``true``
     - ``false``
   * - ``rollout.n``
     - ``4`` (multiple rollouts per prompt)
     - ``1`` (single rollout)
   * - ``critic`` section
     - Not present
     - Required (learned value function)
   * - VRAM usage
     - Lower (no critic)
     - Higher (actor + critic + ref)
