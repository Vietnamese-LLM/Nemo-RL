# Setup Environment Variables
```bash
export HF_TOKEN=<your_huggingface_token>
export WANDB_API_KEY=<your_wandb_api_key>
export TARGET_NODES=2
export NUM_ACTOR_NODES='worker-1,worker-2'
```

# Submit Job
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