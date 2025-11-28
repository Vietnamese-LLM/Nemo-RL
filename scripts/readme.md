# Setup Environment Variables
```bash
export HF_TOKEN=<your_huggingface_token>
export WANDB_API_KEY=<your_wandb_api_key>
export TARGET_NODES=2
export NUM_ACTOR_NODES='worker-1,worker-2'
```

# Submit Job
```bash
sbatch scripts/train_grpo.sh
```