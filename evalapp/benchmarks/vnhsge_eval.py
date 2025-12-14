# vnhsge_eval.py
"""
VNHSGE-V Evaluation (MCQ, không cần judge LLM)
----------------------------------------------

Cấu trúc thư mục:
  VNHSGE-V/
    Biology/
      MET_Bio_IE_2019.json
      MET_Bio_IE_2020.json
      ...
    Chemistry/
      MET_Chem_IE_2019.json
      ...

Mỗi file JSON: list câu hỏi dạng:
{
  "ID": "MET_Bio_IE_2019_1",
  "Image_Question": "",
  "Question": "Câu 81: ...\nA. ...\nB. ...\nC. ...\nD. ...",
  "Choice": "B",  # đáp án đúng
  "Image_Answer": "",
  "Explanation": "..."
}

Yêu cầu:
- Loại các câu hỏi có dùng ảnh:
    Image_Question != "" hoặc Image_Answer != ""
- Model: sinh log-prob cho các đáp án A/B/C/D và chọn argmax.
- Evaluation:
    * accuracy cho từng subject/subset (vd: Biology/MET_Bio_IE_2019)
    * tổng cho từng subject
    * tổng overall
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from tqdm.auto import tqdm


CHOICE_LETTERS = ["A", "B", "C", "D"]


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_eval_dataset(root_dir: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    root_dir = "VNHSGE-V"

    Return:
    {
      "Biology": {
          "MET_Bio_IE_2019": [ {...}, {...}, ... ],
          "MET_Bio_IE_2020": [...],
          ...
      },
      "Chemistry": {
          "MET_Chem_IE_2019": [...],
          ...
      },
      ...
    }
    """
    root = Path(root_dir)
    all_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for subject_dir in root.iterdir():
        if not subject_dir.is_dir():
            continue

        subject_name = subject_dir.name   # "Biology", "Chemistry", ...
        subject_data: Dict[str, List[Dict[str, Any]]] = {}

        for json_path in subject_dir.glob("*.json"):
            key_name = json_path.stem      # "MET_Bio_IE_2019"
            with open(json_path, "r", encoding="utf-8") as f:
                subject_data[key_name] = json.load(f)

        if subject_data:
            all_data[subject_name] = subject_data

    return all_data


# =========================================================
# 2. PROMPT FORMAT & LOGPROB
# =========================================================

def format_vnhsge_prompt(question_text: str) -> str:
    """
    Format prompt cho MCQ.
    question_text đã chứa cả đề + các lựa chọn A/B/C/D.
    Ta chỉ thêm yêu cầu trả lời bằng 1 chữ cái.
    """
    prompt = (
        f"{question_text.strip()}\n\n"
        "Chọn đáp án đúng (A, B, C hoặc D).\n"
        "Đáp án đúng là:"
    )
    return prompt


@torch.no_grad()
def compute_logprob_for_choice(
    model,
    tokenizer,
    prompt: str,
    answer_letter: str,
    max_tokens: int = 512,
) -> float:
    """
    Tính log P(answer_letter | prompt), tương tự MMLU.
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


def predict_vnhsge_single(
    model,
    tokenizer,
    question_text: str,
) -> str:
    """
    Trả về chữ cái đáp án dự đoán ("A"/"B"/"C"/"D").
    """
    prompt = format_vnhsge_prompt(question_text)
    scores = []
    for letter in CHOICE_LETTERS:
        lp = compute_logprob_for_choice(model, tokenizer, prompt, letter)
        scores.append(lp)

    scores_t = torch.tensor(scores)
    idx = int(scores_t.argmax().item())
    return CHOICE_LETTERS[idx]


# =========================================================
# 3. FULL EVALUATION
# =========================================================

def run_vnhsge_eval(
    model,
    tokenizer,
    root_dir: str = "./VNHSGE-V",
    max_samples_per_subset: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate toàn bộ VNHSGE-V.

    - model, tokenizer: LLM dạng causal (Qwen3-4B, Qwen2.5, ...)
    - root_dir: thư mục chứa các subject folder.
    - max_samples_per_subset: nếu không None, giới hạn số câu hỏi
                              đầu tiên trong mỗi subset để chạy nhanh.

    Trả về dict kết quả dạng:

    {
      "per_subject_subset": {
          "Biology": {
              "MET_Bio_IE_2019": {"accuracy": 0.75, "num_questions": 40},
              "MET_Bio_IE_2020": {...},
              ...
          },
          "Chemistry": {...},
          ...
      },
      "per_subject_overall": {
          "Biology": {"accuracy": 0.78, "num_questions": 120},
          "Chemistry": {...},
          ...
      },
      "overall": {
          "accuracy": 0.80,
          "num_questions": 600
      }
    }
    """
    all_eval = load_eval_dataset(root_dir)

    per_subject_subset: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_subject_overall: Dict[str, Dict[str, float]] = {}

    total_correct_all = 0
    total_count_all = 0

    device = next(model.parameters()).device
    print(f"[vnhsge_eval] Using device: {device}")

    for subject_name, subject_data in all_eval.items():
        subject_results: Dict[str, Dict[str, float]] = {}
        subject_correct = 0
        subject_count = 0

        for subset_name, questions in subject_data.items():
            # Filter bỏ câu có ảnh
            filtered = [
                q for q in questions
                if (not q.get("Image_Question")) and (not q.get("Image_Answer"))
            ]

            if max_samples_per_subset is not None and max_samples_per_subset > 0:
                filtered = filtered[: max_samples_per_subset]

            if not filtered:
                # không có câu hỏi hợp lệ trong subset này
                continue

            correct = 0
            total = 0

            for q in tqdm(filtered, desc=f"[{subject_name}/{subset_name}]", leave=False):
                question_text = q["Question"]
                gold_choice = q["Choice"].strip()

                # Bỏ trường hợp gold không thuộc A/B/C/D
                if gold_choice not in CHOICE_LETTERS:
                    continue

                pred_choice = predict_vnhsge_single(model, tokenizer, question_text)

                if pred_choice == gold_choice:
                    correct += 1
                total += 1

            if total == 0:
                continue

            acc = correct / total
            subject_results[subset_name] = {
                "accuracy": acc,
                "num_questions": total,
            }

            subject_correct += correct
            subject_count += total

        if subject_results:
            per_subject_subset[subject_name] = subject_results

        if subject_count > 0:
            per_subject_overall[subject_name] = {
                "accuracy": subject_correct / subject_count,
                "num_questions": subject_count,
            }

        total_correct_all += subject_correct
        total_count_all += subject_count

    overall = {
        "accuracy": (total_correct_all / total_count_all) if total_count_all > 0 else math.nan,
        "num_questions": total_count_all,
    }

    return {
        "per_subject_subset": per_subject_subset,
        "per_subject_overall": per_subject_overall,
        "overall": overall,
    }
