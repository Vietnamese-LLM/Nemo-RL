#!/bin/bash

# Exit the script if any command fails
set -eoux pipefail

# === 1. Slurm Parameters ===
SLURM_ACCOUNT="root"
SLURM_PARTITION="main"
JOB_NAME="grpo-llama8b-2nodes"
CONTAINER_IMAGE="docker://ghcr.io/elfsong/nemo-rl:latest"

# === 2. Environment Variables ===
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}
export NUM_ACTOR_NODES=${NUM_ACTOR_NODES}

# === 3. Mount Disk ===
HF_CACHE_DIR="$HOME/.cache/huggingface"
mkdir -p $HF_CACHE_DIR
MOUNTS="$PWD:$PWD,$HF_CACHE_DIR:$HF_CACHE_DIR"

# === 4. Command ===
COMMAND="export HF_TOKEN=$HF_TOKEN && \
    export WANDB_API_KEY=$WANDB_API_KEY && \
	uv run ./examples/run_grpo_math.py \
    --config examples/configs/grpo_math_8B.yaml \
    cluster.num_nodes=${NUM_ACTOR_NODES} \
    checkpointing.checkpoint_dir='results/llama8b_${NUM_ACTOR_NODES}nodes' \
    logger.wandb_enabled=True \
    logger.wandb.name='${JOB_NAME}'"

# === 5. Parameter Checks ===
echo "==========================================================="
echo "[+] Job Name: ${JOB_NAME}"
echo "-----------------------------------------------------------"

if [ -z "$HF_TOKEN" ]; then
    echo "[-] Error: HF_TOKEN is not set"
    exit 1
fi
if [ -z "$WANDB_API_KEY" ]; then
    echo "[-] Error: WANDB_API_KEY is not set"
    exit 1
fi
if [ -z "$MOUNTS" ]; then
    echo "[-] Error: MOUNTS is not set"
    exit 1
fi
if [ -z "$COMMAND" ]; then
    echo "[-] Error: COMMAND is not set"
    exit 1
fi

echo "[+] Slurm Account: ${SLURM_ACCOUNT}"
echo "[+] Slurm Partition: ${SLURM_PARTITION}"
echo "[+] Number of Actor Nodes: ${NUM_ACTOR_NODES}"
echo "[+] Container Image: ${CONTAINER_IMAGE}"
echo "[+] HF_TOKEN: ${HF_TOKEN}"
echo "[+] WANDB_API_KEY: ${WANDB_API_KEY}"
echo "[+] MOUNTS: ${MOUNTS}"
echo "[+] COMMAND: ${COMMAND}"
echo "==========================================================="

# === 5. Submit ===
sbatch \
        --nodes=${NUM_ACTOR_NODES} \
        --account=${SLURM_ACCOUNT} \
        --job-name=${JOB_NAME} \
        --partition=${SLURM_PARTITION} \
        --exclusive \
        --mem=0 \
        --time=4:0:0 \
        --gres=gpu:8 \
        --export=ALL,COMMAND="$COMMAND",CONTAINER="$CONTAINER_IMAGE",MOUNTS="$MOUNTS" \
        ray.sub