# model_utils.py
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hai lựa chọn model chính
MODEL_REGISTRY: Dict[str, str] = {
    "Qwen3-32B": "weights/Qwen3-32B",
    "Qwen3-4B": "weights/Qwen3-4B",
    "Qwen3-4B-Instruct": "weights/Qwen3-4B-Instruct-2507",
}

# Cache model
_CURRENT_MODEL = None
_CURRENT_TOKENIZER = None
_CURRENT_MODEL_ID = None


def load_qwen_model(
    model_key: str,
    checkpoint_path: Optional[str] = None,
):
    """
    Load Qwen model theo:
      - model_key: "Qwen3-4B" hoặc "Qwen3-4B-Instruct"
      - checkpoint_path:
          + None: dùng HF ID trong MODEL_REGISTRY
          + path: folder chứa weight đã save (một step nào đó)
    """
    global _CURRENT_MODEL, _CURRENT_TOKENIZER, _CURRENT_MODEL_ID

    if model_key not in MODEL_REGISTRY and checkpoint_path is None:
        raise ValueError(f"Unknown model_key: {model_key}")
    
    print("[model_utils] Loading model from: ", model_key, checkpoint_path)
    base_id = MODEL_REGISTRY.get(model_key, None)
    model_path = checkpoint_path if checkpoint_path is not None else base_id
    cache_id = f"{model_key}::{model_path}"

    if _CURRENT_MODEL is not None and _CURRENT_MODEL_ID == cache_id:
        print(f"[model_utils] Reuse cached model: {cache_id}")
        return _CURRENT_MODEL, _CURRENT_TOKENIZER, model_path

    print(f"[model_utils] Loading model and tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.bfloat16,
        device_map="auto",
        # trust_remote_code=True,
    )
    print("[model_utils] Model and tokenizer loaded successfully")
    model.eval()

    _CURRENT_MODEL = model
    _CURRENT_TOKENIZER = tokenizer
    _CURRENT_MODEL_ID = cache_id
    return model, tokenizer, model_path


def get_current_model_and_tokenizer():
    """
    Get current model and tokenizer from cache.
    """
    global _CURRENT_MODEL, _CURRENT_TOKENIZER
    return _CURRENT_MODEL, _CURRENT_TOKENIZER


def list_checkpoints_in_folder(
    weights_root: str,
    pattern: str = r"(?:step|ckpt)[_-]?(\d+)",
) -> List[Tuple[int, str]]:
    """
    Scan folder 'weights_root', find subfolders or files containing step numbers,
    e.g: step_000500, step-20000, ckpt_30000, ckpt-40000, ...
    Return list (step_int, checkpoint_path) sorted by step in ascending order.

    Assumption: each subfolder is a checkpoint that can be loaded by from_pretrained.
    """
    if not os.path.isdir(weights_root):
        raise ValueError(f"{weights_root} is not a directory")

    rx = re.compile(pattern)
    candidates: List[Tuple[int, str]] = []

    for name in os.listdir(weights_root):
        full_path = os.path.join(weights_root, name)
        m = rx.search(name)
        if not m:
            continue

        try:
            step = int(m.group(1))
        except ValueError:
            continue

        # Prefer subfolder; if file, still add it (you can organize it as you want)
        if os.path.isdir(full_path) or os.path.isfile(full_path):
            candidates.append((step, full_path))

    candidates.sort(key=lambda x: x[0])
    return candidates
