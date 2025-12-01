#!/bin/bash

# Exit the script if any command fails
set -eoux pipefail

# === 1. Slurm Parameters ===
SLURM_ACCOUNT="root"
SLURM_PARTITION="main"
CONTAINER_IMAGE="docker://ghcr.io/elfsong/nemo-rl:latest"

# === 2. Environment Variables ===
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}
export NUM_ACTOR_NODES=1
export TARGET_NODES=${TARGET_NODES}

JOB_NAME="dpo-multiple-node-${NUM_ACTOR_NODES}"

# === 3. Mount Disk ===
HF_CACHE_DIR="$HOME/.cache/huggingface"
mkdir -p $HF_CACHE_DIR
MOUNTS="$PWD:$PWD,$HF_CACHE_DIR:$HF_CACHE_DIR,/dev/infiniband:/dev/infiniband"

# === 4. Command ===
COMMAND="export HF_TOKEN=$HF_TOKEN && \
    export WANDB_API_KEY=$WANDB_API_KEY && \
    uv run python examples/run_dpo.py \
    policy.model_name="meta-llama/Llama-3.1-8B-Instruct" \
    policy.precision="bfloat16" \
    policy.train_global_batch_size=256 \
    checkpointing.checkpoint_dir='results/${JOB_NAME}' \
    cluster.gpus_per_node=8 \
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
        --time=1:0:0 \
        --gres=gpu:8 \
        --export=ALL,COMMAND="$COMMAND",CONTAINER="$CONTAINER_IMAGE",MOUNTS="$MOUNTS" \
        ray.sub