# vmlu_eval.py
"""
VMLU Evaluation (MCQ, no judge)
--------------------------------

Dataset: ./VMLU/valid.jsonl

Mỗi dòng là một JSON:
{
  "id": "28-0007",
  "question": "Hoạt động nào sau đây ...",
  "choices": [
      "A. ...",
      "B. ...",
      "C. ...",
      "D. ..."
  ],
  "answer": "C"
}

Ý tưởng:
- Format prompt: question + các lựa chọn + hướng dẫn "Đáp án đúng là:"
- Tính log P(letter | prompt) cho từng letter tương ứng với lựa chọn (A/B/C/D).
- Chọn argmax -> so sánh với `answer`.
- Trả về overall accuracy.
"""

import json
import math
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm.auto import tqdm

CHOICE_LETTERS = ["A", "B", "C", "D"]

_VMLU_CACHE: Dict[str, List[Dict[str, Any]]] = {}


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_vmlu_dataset(
    jsonl_path: str = "./VMLU/valid.jsonl",
) -> List[Dict[str, Any]]:
    """
    Load VMLU từ file JSONL và cache theo path.
    """
    global _VMLU_CACHE
    if jsonl_path in _VMLU_CACHE:
        return _VMLU_CACHE[jsonl_path]

    data: List[Dict[str, Any]] = []
    print(f"[vmlu_eval] Loading VMLU from {jsonl_path} ...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    _VMLU_CACHE[jsonl_path] = data
    return data


# =========================================================
# 2. PROMPT & LOGPROB
# =========================================================

def format_vmlu_prompt(question: str, choices: List[str]) -> str:
    """
    Prompt cho MCQ VMLU.

    choices đã là dạng ["A. ...", "B. ...", ...],
    ta chỉ cần nối vào question và thêm hướng dẫn.
    """
    lines = [question.strip()]
    for c in choices:
        lines.append(c.strip())
    lines.append("")
    lines.append("Chọn đáp án đúng (A, B, C hoặc D).")
    lines.append("Đáp án đúng là:")
    return "\n".join(lines)


@torch.no_grad()
def compute_logprob_for_choice(
    model,
    tokenizer,
    prompt: str,
    answer_letter: str,
    max_tokens: int = 512,
) -> float:
    """
    Tính log P(answer_letter | prompt).
    """
    device = next(model.parameters()).device

    full_text = prompt + " " + answer_letter
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = enc["input_ids"].to(device)

    ans_ids = tokenizer(answer_letter, add_special_tokens=False)["input_ids"]
    ans_len = len(ans_ids)

    outputs = model(input_ids)
    logits = outputs.logits  # [1, L, V]
    seq_len = input_ids.size(1)

    start = seq_len - ans_len
    logits_ans = logits[:, start - 1 : seq_len - 1, :]  # [1, ans_len, V]
    log_probs = torch.log_softmax(logits_ans, dim=-1)

    ans_tensor = torch.tensor(ans_ids, dtype=torch.long, device=device)
    ans_tensor = ans_tensor.view(1, ans_len, 1)  # [1, ans_len, 1]

    token_logprobs = log_probs.gather(-1, ans_tensor).squeeze(-1)  # [1, ans_len]
    token_logprobs = token_logprobs.sum(dim=1)  # [1]
    return float(token_logprobs.item())


def predict_vmlu_single(
    model,
    tokenizer,
    question: str,
    choices: List[str],
) -> str:
    """
    Trả về chữ cái đáp án dự đoán: "A"/"B"/"C"/"D".
    """
    prompt = format_vmlu_prompt(question, choices)

    # số lựa chọn thực tế (có thể < 4, nhưng default là 4)
    num_choices = min(len(choices), len(CHOICE_LETTERS))

    scores = []
    for i in range(num_choices):
        letter = CHOICE_LETTERS[i]
        lp = compute_logprob_for_choice(model, tokenizer, prompt, letter)
        scores.append(lp)

    scores_t = torch.tensor(scores)
    idx = int(scores_t.argmax().item())
    return CHOICE_LETTERS[idx]


# =========================================================
# 3. FULL EVALUATION
# =========================================================

def run_vmlu_eval(
    model,
    tokenizer,
    jsonl_path: str = "./VMLU/valid.jsonl",
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Evaluate model trên toàn bộ (hoặc subset) VMLU.

    Trả về:
        acc_by_subject = {"overall": accuracy}
        count_by_subject = {"overall": n}
    -> cùng style với các benchmark MCQ khác.
    """
    data = load_vmlu_dataset(jsonl_path)

    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]

    total = 0
    correct = 0

    for item in tqdm(data, desc="[VMLU] Evaluating", leave=False):
        question = item["question"]
        choices = item["choices"]
        gold = item["answer"].strip()

        # phòng trường hợp answer không thuộc A..D
        if gold not in CHOICE_LETTERS:
            continue

        pred = predict_vmlu_single(model, tokenizer, question, choices)

        if pred == gold:
            correct += 1
        total += 1

    if total > 0:
        acc = correct / total
    else:
        acc = math.nan

    acc_by_subject = {"overall": acc}
    count_by_subject = {"overall": total}
    return acc_by_subject, count_by_subject
