.. _changelog-page:

=========
Changelog
=========

2026-02-26: Experiment Framework
---------------------------------

**Added:**

- Config-driven experiment framework under ``experiments/``
- YAML base configs for 4 model/algorithm combinations:

  - ``grpo_qwen3_1.7b`` — Qwen/Qwen3-1.7B with GRPO
  - ``grpo_deepseek_1.5b`` — DeepSeek-R1-Distill-Qwen-1.5B with GRPO
  - ``ppo_qwen3_1.7b`` — Qwen/Qwen3-1.7B with PPO
  - ``ppo_deepseek_1.5b`` — DeepSeek-R1-Distill-Qwen-1.5B with PPO

- Sweep configs for learning rate and KL coefficient search
- ``parse_config.py`` — YAML parser that outputs verl CLI args, W&B JSON, and
  log metadata headers
- ``train.sh`` — universal single-experiment launcher with ``--dry_run`` support
- ``sweep.sh`` — hyperparameter sweep orchestrator with GPU queue management
- ``self_evolve.sh`` — multi-round self-evolving RL training loop
- ``run_experiments.sh`` — root-level shortcut with auto-expanding config names
- 3-layer experiment tracking: git (configs), local logs (with metadata header),
  W&B (real-time metrics)
- Explicit GPU management — only user-specified GPUs via ``--gpus``, no
  auto-discovery

**Removed:**

- ``train_qwen3_4b_single_gpu.sh`` — replaced by ``train.sh`` + YAML configs
- ``run_self_evolve_rl.sh`` — replaced by ``self_evolve.sh``
- ``train_grpo_math_tune_ray.sh`` — replaced by ``train.sh`` + YAML configs

**Design docs:**

- ``docs/plans/2026-02-26-experiment-framework-design.md``
- ``docs/plans/2026-02-26-experiment-framework-impl.md``
