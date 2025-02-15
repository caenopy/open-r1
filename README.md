# Train option parser with RL

```
sudo apt-get install git-lfs
git lfs pull

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip --link-mode=copy

uv pip install vllm==0.7.1 --link-mode=copy
uv pip install setuptools hnswlib peft
uv pip install -e .
uv pip install flash_attn --no-build-isolation

huggingface-cli login
uv pip install wandb
wandb login
```

Export the following: `HF_TOKEN`, `OPENAI_API_KEY`, `CO_API_KEY`, `PERPLEXITY_API_KEY`

To start GRPO training:
```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml     --num_processes=7 src/open_r1/grpo.py     --config recipes/Llama3.1-8B-Instruct/grpo/intent_to_skill_rl.yaml
```