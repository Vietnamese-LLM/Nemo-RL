#!/bin/bash

# Exit the script if any command fails
set -eoux pipefail

# === 1. Slurm Parameters ===
SLURM_ACCOUNT="root"
SLURM_PARTITION="main"
# CONTAINER_IMAGE="docker://ghcr.io/elfsong/nemo-rl:latest"
CONTAINER_IMAGE="./docker/elfsong+nemo-rl+latest.sqsh"

# === 2. Environment Variables ===
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}
export NUM_ACTOR_NODES=${NUM_ACTOR_NODES}
export TARGET_NODES=${TARGET_NODES}
export JOB_NAME="dpo-nodes-${NUM_ACTOR_NODES}-qwen3-32b"

# === 3. Mount Disk ===
HF_CACHE_DIR="$HOME/.cache/huggingface"
mkdir -p $HF_CACHE_DIR
MOUNTS="$PWD:$PWD,$HF_CACHE_DIR:$HF_CACHE_DIR,/dev/infiniband:/dev/infiniband"

# === 4. Command ===
COMMAND="export HF_TOKEN=$HF_TOKEN && \
    export WANDB_API_KEY=$WANDB_API_KEY && \
    uv run examples/run_dpo.py \
    --config examples/configs/dpo.yaml \
    cluster.num_nodes=${NUM_ACTOR_NODES} \
    cluster.gpus_per_node=8 \
    policy.precision="bfloat16" \
    policy.dtensor_cfg.tensor_parallel_size=8 \
    policy.dtensor_cfg.activation_checkpointing=false \
    policy.train_micro_batch_size=1 \
    policy.max_total_sequence_length=4096 \
    dpo.val_global_batch_size=32 \
    checkpointing.checkpoint_dir='results/${JOB_NAME}' \
    logger.wandb_enabled=True \
    logger.wandb.name='${JOB_NAME}'"

# === 5. Parameter Checks ===
echo "Job Name: ${JOB_NAME}"

# === 5. Submit ===
sbatch \
        --nodelist=${TARGET_NODES} \
        --nodes=${NUM_ACTOR_NODES} \
        --account=${SLURM_ACCOUNT} \
        --job-name=${JOB_NAME} \
        --partition=${SLURM_PARTITION} \
        --exclusive \
        --mem=0 \
        --time=6:0:0 \
        --gres=gpu:8 \
        --export=ALL,COMMAND="$COMMAND",CONTAINER="$CONTAINER_IMAGE",MOUNTS="$MOUNTS" \
        ray.sub