#!/bin/bash
# =============================================================================
# Train Qwen3-4B with GRPO on a single GPU (e.g., RTX 4090 24GB)
#
# This script adapts the original multi-GPU Ray-based training to run on a
# single GPU. Key adaptations for 4090 (24GB VRAM):
#   - Use HF rollout (not vllm, since vllm 0.6.3 doesn't support Qwen3)
#   - Enable CPU offloading for optimizer and ref model params
#   - Smaller batch sizes to fit in memory
#   - rollout_tp=1 (single GPU, no tensor parallelism)
#   - Gradient checkpointing enabled
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash train_qwen3_4b_single_gpu.sh [OPTIONS]
#
# Required env vars:
#   MODEL_PATH      - Path to Qwen3-4B model weights (local or HF hub)
#   DATA_DIR        - Path to directory containing train.parquet and test.parquet
#
# Optional env vars:
#   WANDB_API_KEY   - For logging to W&B
#   OUTPUT_DIR      - Checkpoint output directory (default: ./checkpoints)
# =============================================================================

set -x

# ---- Environment ----
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND=XFORMERS

# ---- Defaults (overridable via env vars or CLI args) ----
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-./data/simplelr_qwen_level3to5}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---- Hyperparameters tuned for single 4090 (24GB) ----
# Batch sizes: small to fit in 24GB
TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE=1
MICRO_ROLLOUT_BATCH_SIZE=4

# Sequence lengths
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=3072

# GRPO specific
ROLLOUT_N=4
LEARNING_RATE=1e-6
KL_LOSS_COEF=0.0001
KL_COEF=0.001
ENTROPY_COEFF=0.001
CLIP_RATIO=0.2
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=0.6

# Training schedule
TOTAL_EPOCHS=20
SAVE_FREQ=5
TEST_FREQ=5

# Rollout config (HF-based for Qwen3 compatibility)
ROLLOUT_NAME=hf
ROLLOUT_GPU_MEMORY_UTIL=0.4
LOG_PROB_MICRO_BATCH_SIZE=2

# ---- Parse CLI arguments ----
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
        --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
        --entropy_coeff) ENTROPY_COEFF="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --save_freq) SAVE_FREQ="$2"; shift 2 ;;
        --rollout_name) ROLLOUT_NAME="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---- Derived values ----
MODEL_NAME=$(basename "$MODEL_PATH")
RUN_NAME="verl-grpo_${MODEL_NAME}_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}_lr${LEARNING_RATE}"
if [ -n "$SUFFIX" ]; then
    RUN_NAME="${RUN_NAME}_${SUFFIX}"
fi
CKPT_DIR="${OUTPUT_DIR}/${RUN_NAME}"

echo "============================================="
echo "Training Qwen3-4B on Single GPU"
echo "============================================="
echo "Model:             $MODEL_PATH"
echo "Data:              $DATA_DIR"
echo "Run name:          $RUN_NAME"
echo "Checkpoint dir:    $CKPT_DIR"
echo "Batch size:        $TRAIN_BATCH_SIZE"
echo "Rollout N:         $ROLLOUT_N"
echo "Max prompt len:    $MAX_PROMPT_LENGTH"
echo "Max response len:  $MAX_RESPONSE_LENGTH"
echo "Learning rate:     $LEARNING_RATE"
echo "KL loss coef:      $KL_LOSS_COEF"
echo "Temperature:       $TEMPERATURE"
echo "Rollout engine:    $ROLLOUT_NAME"
echo "Total epochs:      $TOTAL_EPOCHS"
echo "============================================="

# ---- Launch Ray (single node, single GPU) ----
ray stop --force 2>/dev/null
ray start --head --num-gpus 1 --num-cpus 8

sleep 3

# ---- Run GRPO training ----
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=50 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.micro_rollout_batch_size=$MICRO_ROLLOUT_BATCH_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=qwen3_grpo \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.remove_previous_ckpt=False 2>&1 | tee "${CKPT_DIR}.log"
