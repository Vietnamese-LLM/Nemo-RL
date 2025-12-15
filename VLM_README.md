# VLM Post-training

## Step 1: Prerequisites

```bash
# 1.1 Clone the repository
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl

# 1.2 Login to GitHub Container Registry
export CR_PAT=<your_github_personal_access_token>      
echo $CR_PAT | docker login ghcr.io -u <your_github_username> --password-stdin

# 1.3 Build the Docker image
docker buildx build --platform linux/amd64 --target release -f docker/Dockerfile --tag ghcr.io/elfsong/nemo-rl:latest --push .

# 1.4.1 Set Enroot Credentials (Option 1, Slow)
mkdir -p ~/.config/enroot

cat <<EOF > ~/.config/enroot/.credentials
machine ghcr.io login <your_github_username> password <your_github_personal_access_token>
EOF

chmod 600 ~/.config/enroot/.credentials

# 1.4.2 Save Docker sqsh (Option 2, Fast)
docker save ghcr.io/<your_github_username>/nemo-rl:latest -o nemo-rl.sqsh
```

## Step 2: Training

```bash
# 2.1 Set Environment Variables
export HF_TOKEN=<your_huggingface_personal_access_token>
export WANDB_API_KEY=<your_wandb_api_key>
export NUM_ACTOR_NODES=<number_of_nodes>
export TARGET_NODES="<node_list>"

# 2.2 Submit the Job
# For DPO multiple nodes training
bash scripts/train_dpo_multiple_node.sh
# For GRPO multiple nodes training
bash scripts/train_grpo_multiple_node.sh

# To modify the training config, please refer to the following files:
# - examples/configs/dpo.yaml
# - examples/configs/grpo.yaml
```

## Step 3: Evaluation

```bash
srun --gres=gpu:8 -c 32 --time=04:00:00 --mem=256G --pty bash -i

CHECKPOINT_PATH='<your_checkpoint_path>'

# 3.1 Convert the Model to Hugging Face Format
uv run python examples/converters/convert_dcp_to_hf.py \
    --config ${CHECKPOINT_PATH}/config.yaml \
    --dcp-ckpt-path ${CHECKPOINT_PATH}/policy/weights/ \
    --hf-ckpt-path ${CHECKPOINT_PATH}/hf

# 3.2 Evaluate the Model via EvalApp
python evalapp/main_app.py
```

