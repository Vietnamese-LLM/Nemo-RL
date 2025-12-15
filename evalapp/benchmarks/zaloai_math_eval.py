# zalo_math_eval.py
"""
Zalo AI Elementary Math Benchmark Evaluation
--------------------------------------------
Benchmark format:
[
  {
    "id": "...",
    "question": "...",
    "choices": ["A. ...", "B. ...", "C. ...", "D. ..."]
  },
  ...
]

Evaluation rule:
- Chọn lựa chọn A/B/C/D có log-probability cao nhất.
- Trả về accuracy tổng thể.
"""

import json
import math
from typing import Dict, List, Tuple, Optional

import torch
from tqdm.auto import tqdm

# ------------------------------------------
# CONSTANTS
# ------------------------------------------

CHOICE_LETTERS = ["A", "B", "C", "D"]

# Cache dataset
_ZALO_MATH_CACHE = None


# ------------------------------------------
# DATA LOADING
# ------------------------------------------

def load_zalo_math_dataset(json_path: str):
    """
    Load Zalo Math test set từ file JSON
    Format:
      { "data": [ {id, question, choices}, ... ] }
    Cache global để không load lại mỗi lần.
    """
    global _ZALO_MATH_CACHE
    if _ZALO_MATH_CACHE is not None:
        return _ZALO_MATH_CACHE

    print(f"[zalo_math_eval] Loading dataset from {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = raw["data"]
    _ZALO_MATH_CACHE = dataset
    return dataset


# ------------------------------------------
# PROMPT FORMAT
# ------------------------------------------

def format_zalo_math_prompt(question: str, choices: List[str]) -> str:
    """
    Format prompt tương tự MMLU:
    Câu hỏi
    A. ...
    B. ...
    C. ...
    D. ...
    Answer:
    """
    lines = [question.strip()]
    for c in choices:
        lines.append(c)  # đã có "A. ...", "B. ..." sẵn trong data
    lines.append("\nAnswer:")
    return "\n".join(lines)


# ------------------------------------------
# LOG PROBABILITY COMPUTATION
# ------------------------------------------

@torch.no_grad()
def compute_logprob_for_choice(
    model,
    tokenizer,
    prompt: str,
    answer_letter: str,
    max_tokens: int = 512,
) -> float:
    """
    Tính log-probability P(answer_letter | prompt).
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

    # Token IDs của đáp án (chỉ 1 ký tự: 'A', 'B', ...)
    ans_ids = tokenizer(answer_letter, add_special_tokens=False)["input_ids"]
    ans_len = len(ans_ids)

    outputs = model(input_ids)
    logits = outputs.logits  # [1, L, V]
    seq_len = input_ids.size(1)

    # vị trí token của answer trong chuỗi
    start = seq_len - ans_len
    logits_ans = logits[:, start - 1 : seq_len - 1, :]  # [1, ans_len, vocab]
    log_probs = torch.log_softmax(logits_ans, dim=-1)

    ans_tensor = torch.tensor(ans_ids, dtype=torch.long, device=device)
    ans_tensor = ans_tensor.view(1, ans_len, 1)  # [1, ans_len, 1]

    token_logprobs = log_probs.gather(-1, ans_tensor).squeeze(-1)
    token_logprobs = token_logprobs.sum(dim=1)  # tổng log prob
    return float(token_logprobs.item())


def predict_zalo_math_single(
    model,
    tokenizer,
    question: str,
    choices: List[str],
) -> int:
    """
    Trả về index 0..3 của lựa chọn có log-prob cao nhất.
    """
    # Lấy letter từ dòng "A. xxx"
    # choices: ["A. ...", "B. ...", "C. ...", "D. ..."]
    letters = [c.split(".", 1)[0].strip() for c in choices]

    prompt = format_zalo_math_prompt(question, choices)
    scores = []

    for letter in letters:
        lp = compute_logprob_for_choice(model, tokenizer, prompt, letter)
        scores.append(lp)

    scores_t = torch.tensor(scores)
    return int(scores_t.argmax().item())


# ------------------------------------------
# FULL BENCHMARK EVALUATION
# ------------------------------------------

def run_zalo_math_eval(
    model,
    tokenizer,
    json_path: str = "./Elementary-Math-Solving-Zalo-AI-2023/datasets/math_test_with_hand_label.json",
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Evaluate toàn bộ Zalo Math test set.
    Trả về:
        - acc_by_subject (1 key: "overall")
        - count_by_subject (1 key: "overall")
    """

    dataset = load_zalo_math_dataset(json_path)

    if max_samples is not None and max_samples > 0:
        dataset = dataset[:max_samples]

    total = len(dataset)
    correct = 0

    for item in tqdm(dataset, desc="[ZaloMath] Evaluating", leave=False):
        question = item["question"]
        choices = item["choices"]
        gold = item.get("answer", None)

        # Nếu dataset không có trường "answer"
        # => không thể evaluate accuracy
        # => skip
        if gold is None:
            continue

        # gold là chữ cái: "A", "B", "C", "D"
        pred_idx = predict_zalo_math_single(model, tokenizer, question, choices)

        pred_letter = CHOICE_LETTERS[pred_idx]
        if pred_letter == gold[0]:
            correct += 1

    if total > 0:
        overall_acc = correct / total
    else:
        overall_acc = math.nan

    return {"overall": overall_acc}, {"overall": total}
