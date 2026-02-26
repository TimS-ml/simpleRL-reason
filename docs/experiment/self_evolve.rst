.. _self-evolve-page:

=============================
Self-Evolving RL
=============================

Overview
--------

Self-evolving RL is a multi-round training strategy that combines hyperparameter
search with iterative model improvement:

1. **Round 1**: Run a sweep of HP variants starting from the base model
2. **Selection**: Pick the best checkpoint (by val/correctness)
3. **Round 2**: Run the same sweep starting from the best Round 1 checkpoint
4. **Repeat**: Each round builds on the best result from the previous round

This allows the model to progressively improve while exploring different
hyperparameter regions in each round.

.. code-block:: text

   Round 1: base model ──→ [sweep: lr=5e-7, lr=1e-6, lr=2e-6, lr=5e-6]
                                    │
                           pick best (lr=1e-6, score=0.45)
                                    │
   Round 2: best ckpt  ──→ [sweep: lr=5e-7, lr=1e-6, lr=2e-6, lr=5e-6]
                                    │
                           pick best (lr=2e-6, score=0.62)
                                    │
   Round 3: best ckpt  ──→ [sweep: lr=5e-7, lr=1e-6, lr=2e-6, lr=5e-6]
                                    │
                           pick best (lr=1e-6, score=0.71)
                                    │
                           Final model: score=0.71

Usage
-----

.. code-block:: bash

   bash run_experiments.sh evolve \
       --config grpo_qwen3_1.7b \
       --sweep lr_sweep \
       --gpus 0,1 \
       --rounds 3 \
       --epochs_per_round 10

Or call the script directly:

.. code-block:: bash

   bash experiments/scripts/self_evolve.sh \
       --base_config experiments/configs/base/grpo_qwen3_1.7b.yaml \
       --sweep_config experiments/configs/sweeps/lr_sweep.yaml \
       --gpus 0,1 \
       --rounds 3 \
       --epochs_per_round 10 \
       --output_dir ./checkpoints/self_evolve

Arguments
---------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Argument
     - Default
     - Description
   * - ``--base_config PATH``
     - (required)
     - Path to base YAML config
   * - ``--sweep_config PATH``
     - (required)
     - Path to sweep YAML config
   * - ``--gpus IDS``
     - "0"
     - Comma-separated GPU IDs
   * - ``--rounds N``
     - 3
     - Number of self-evolve rounds
   * - ``--epochs_per_round N``
     - 10
     - Training epochs per round
   * - ``--output_dir PATH``
     - ./checkpoints/self_evolve
     - Output directory

How It Works
------------

**Per round:**

1. Calls ``sweep.sh`` with the current model path and the sweep config
2. Each sweep experiment runs independently on its assigned GPU
3. After all experiments complete, scans log files for ``val/correctness`` scores
4. Selects the checkpoint with the highest score
5. Updates the model path for the next round

**Checkpoint selection:**

- Extracts ``val/correctness`` from training logs using regex
- Takes the maximum score found in each log file
- Compares across all experiments in the round
- If no valid checkpoint is found, continues with the previous model

**Output structure:**

.. code-block:: text

   checkpoints/self_evolve/
   ├── round1/
   │   ├── grpo_qwen3-1.7b_1gpu_lr5e-07_.../
   │   ├── grpo_qwen3-1.7b_1gpu_lr1e-06_.../   ← best
   │   ├── grpo_qwen3-1.7b_1gpu_lr2e-06_.../
   │   └── grpo_qwen3-1.7b_1gpu_lr5e-06_.../
   ├── round2/
   │   ├── ...  (starts from round1 best)
   │   └── ...
   └── round3/
       └── ...


Example Output
--------------

.. code-block:: text

   =============================================
   Self-Evolving RL Training
   =============================================
   Base config:       experiments/configs/base/grpo_qwen3_1.7b.yaml
   Sweep config:      experiments/configs/sweeps/lr_sweep.yaml
   Initial model:     Qwen/Qwen3-1.7B
   GPUs:              0,1
   Rounds:            3
   Epochs per round:  10
   =============================================

   ===== ROUND 1 / 3 =====
   Starting model: Qwen/Qwen3-1.7B
   ...
   Round 1 best: grpo_qwen3-1.7b_lr1e-6 (score: 0.45)
   ===== END ROUND 1 =====

   ===== ROUND 2 / 3 =====
   Starting model: checkpoints/self_evolve/round1/.../global_step_50
   ...
   Round 2 best: grpo_qwen3-1.7b_lr2e-6 (score: 0.62)
   ===== END ROUND 2 =====

   ===== ROUND 3 / 3 =====
   ...

   =============================================
   Self-Evolving RL Complete
   =============================================
   Round-by-round results:
     Round 1: score=0.45
     Round 2: score=0.62
     Round 3: score=0.71
   Final model: checkpoints/self_evolve/round3/.../global_step_50
   =============================================


Tips
----

- **Epochs per round**: Start with 10. Too few may not show meaningful
  differences; too many wastes compute on suboptimal HP.

- **Sweep diversity**: Use sweeps that cover meaningfully different regions.
  The ``lr_sweep`` (4 values spanning 10x range) is a good starting point.

- **Monitoring**: Check W&B during training. If all variants converge to similar
  scores, the sweep range may be too narrow.

- **Disk space**: Each round saves checkpoints for all variants. For a 1.7B
  model with 4 variants per round and 3 rounds, expect ~50-80 GB total.
  Set ``trainer.remove_previous_ckpt: true`` in the config to keep only the
  latest checkpoint per variant.
