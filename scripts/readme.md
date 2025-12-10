# Nemo RL Scripts

## Building Docker Images for Slurm

```bash
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl

export CR_PAT=<your_github_personal_access_token>      
echo $CR_PAT | docker login ghcr.io -u <your_github_username> --password-stdin

# Bulid Self-contained Docker Image
docker buildx build --platform linux/amd64 --target release -f docker/Dockerfile --tag ghcr.io/<your_github_username>/nemo-rl:latest --push .

# Set Enroot Credentials
mkdir -p ~/.config/enroot

cat <<EOF > ~/.config/enroot/.credentials
machine ghcr.io login <your_github_username> password <your_github_personal_access_token>
EOF

chmod 600 ~/.config/enroot/.credentials
```

## Setup Environment Variables

```bash
# 1 node
export HF_TOKEN=<your_huggingface_token>
export WANDB_API_KEY=<your_wandb_api_key>
export NUM_ACTOR_NODES=1
export TARGET_NODES="worker-2"

# 2 nodes
export HF_TOKEN=<your_huggingface_token>
export WANDB_API_KEY=<your_wandb_api_key>
export NUM_ACTOR_NODES=2
export TARGET_NODES="worker-0,worker-1"

# 4 nodes
export HF_TOKEN=<your_huggingface_token>
export WANDB_API_KEY=<your_wandb_api_key>
export NUM_ACTOR_NODES=4
export TARGET_NODES="worker-[0-3]"
```

## Submit Jobs

```bash
# GRPO Single Node
scripts/train_grpo_single_node.sh

# GRPO Multiple Nodes
scripts/train_grpo_multiple_node.sh

# DPO Single Node
scripts/train_dpo_single_node.sh

# DPO Multiple Nodes
scripts/train_dpo_multiple_node.sh
```
