# viquad_eval.py
"""
UIT-ViQuAD2.0 Evaluation with LLM-as-a-Judge
--------------------------------------------

- Dataset: taidng/UIT-ViQuAD2.0 (Vietnamese extractive QA)
- Candidate model: sinh câu trả lời (vd. Qwen3-4B)
- Judge model: Qwen2.5-7B (chạy qua vLLM / OpenAI-compatible API) chấm điểm
  sự tương đồng giữa câu trả lời sinh ra và ground-truth.

Metric:
    - overall_judge_score: trung bình điểm (0..1) do judge đánh giá.
"""

import math
from typing import Dict, Tuple, Optional, List

import json as pyjson
import requests
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

# Cache dataset theo split
_VIQUAD_CACHE = {}  # {split: Dataset}


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_viquad_split(split: str = "validation"):
    """
    Load một split của UIT-ViQuAD2.0 và cache lại.
    """
    global _VIQUAD_CACHE
    if split in _VIQUAD_CACHE:
        return _VIQUAD_CACHE[split]

    print(f"[viquad_eval] Loading UIT-ViQuAD2.0 split='{split}' ...")
    ds = load_dataset("taidng/UIT-ViQuAD2.0")[split]
    _VIQUAD_CACHE[split] = ds
    return ds


# =========================================================
# 2. PROMPT CHO MODEL SINH CÂU TRẢ LỜI
# =========================================================

def format_viquad_prompt(context: str, question: str) -> str:
    """
    Template đơn giản cho QA tiếng Việt.
    Bạn có thể customize cho hợp với style của Qwen3/Qwen2.5.
    """
    prompt = (
        "Đoạn văn sau:\n"
        f"{context.strip()}\n\n"
        "Câu hỏi:\n"
        f"{question.strip()}\n\n"
        "Trả lời ngắn gọn và chính xác:"
    )
    return prompt


@torch.no_grad()
def generate_viquad_answer(
    model,
    tokenizer,
    context: str,
    question: str,
    max_new_tokens: int = 64,
) -> str:
    """
    Sinh câu trả lời từ model ứng viên (candidate).
    """
    device = next(model.parameters()).device
    prompt = format_viquad_prompt(context, question)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = enc["input_ids"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        use_cache=True,  # Enable KV cache for faster generation
        pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
    )

    gen_ids = output_ids[0, input_ids.size(1):]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return answer.strip()


@torch.no_grad()
def generate_viquad_answers_batched(
    model,
    tokenizer,
    contexts: List[str],
    questions: List[str],
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> List[str]:
    """
    Sinh câu trả lời cho nhiều câu hỏi cùng lúc (batching).
    Đây là optimization quan trọng để tăng tốc inference.
    
    Args:
        contexts: List of context strings
        questions: List of question strings
        max_new_tokens: Maximum tokens to generate
        batch_size: Number of samples to process in parallel
    
    Returns:
        List of generated answers
    """
    device = next(model.parameters()).device
    all_answers = []
    
    # Process in batches
    for i in range(0, len(contexts), batch_size):
        batch_contexts = contexts[i:i + batch_size]
        batch_questions = questions[i:i + batch_size]
        
        # Format prompts for batch
        batch_prompts = [
            format_viquad_prompt(ctx, q) 
            for ctx, q in zip(batch_contexts, batch_questions)
        ]
        
        # Tokenize batch with padding
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        # Generate for batch
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            use_cache=True,  # Enable KV cache
            pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
        )
        
        # Decode each answer
        input_lengths = attention_mask.sum(dim=1)
        for j in range(len(batch_prompts)):
            gen_ids = output_ids[j, input_lengths[j]:]
            answer = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_answers.append(answer.strip())
    
    return all_answers


# =========================================================
# 3. LLM-AS-A-JUDGE (Qwen2.5-7B via vLLM API)
# =========================================================

def build_judge_prompt(context: str, question: str, gold_answers, pred_answer: str) -> str:
    """
    Tạo prompt cho judge model.
    Yêu cầu judge trả về một số thực trong [0, 1] (similarity score).
    """
    if isinstance(gold_answers, dict):  # dạng HF answers field
        gold_list = gold_answers.get("text", [])
    else:
        gold_list = gold_answers

    gold_joined = "\n- ".join(gold_list) if gold_list else "(không có)"

    prompt = f"""Bạn là trợ lý đánh giá câu trả lời tiếng Việt cho bài toán hỏi đáp đọc hiểu.

Nhiệm vụ của bạn:
- Đọc đoạn văn, câu hỏi, danh sách câu trả lời tham chiếu (ground truth) và câu trả lời của mô hình.
- Chấm điểm ĐỘ ĐÚNG NGHĨA của câu trả lời mô hình so với các câu trả lời tham chiếu.
- Cho điểm từ 0 đến 1:
  - 1.0: nghĩa tương đương hoặc chỉ khác biệt nhỏ, chấp nhận được.
  - 0.5: đúng một phần, còn thiếu hoặc hơi sai.
  - 0.0: sai hoàn toàn, không liên quan, hoặc mâu thuẫn.
- Chỉ xuất ra MỘT SỐ thực duy nhất trong khoảng [0, 1], với tối đa 2 chữ số sau dấu phẩy.
- Không giải thích thêm, không in text nào khác.

Đoạn văn:
{context}

Câu hỏi:
{question}

Các câu trả lời tham chiếu:
- {gold_joined}

Câu trả lời của mô hình:
{pred_answer}

Hãy cho điểm (0 đến 1) mức độ đúng nghĩa của câu trả lời mô hình so với các câu trả lời tham chiếu.
Chỉ in duy nhất một số thực trong khoảng [0, 1].
"""
    return prompt


def call_judge_api(
    prompt: str,
    judge_base_url: str = "http://localhost:8000/v1",
    judge_api_key: str = "token-abc123",
    judge_model: str = "Qwen2.5-7B-Instruct",
    timeout: int = 60,
) -> float:
    """
    Gọi judge model (Qwen2.5-7B) qua OpenAI-compatible /v1/chat/completions của vLLM.

    YÊU CẦU: vLLM phải được khởi chạy kiểu:
        vllm serve "Qwen/Qwen2.5-7B-Instruct" --api-key token-abc123 --port 8000

    Hàm này:
      - gửi prompt dạng chat
      - nhận về content (string) -> parse float score (0..1)
    """
    url = judge_base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {judge_api_key}",
    }

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"[viquad_eval] Judge API error: {e}")
        return 0.0

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[viquad_eval] Failed to parse judge response JSON: {e}")
        return 0.0

    # cố gắng parse float từ content
    try:
        # nếu judge in thêm text, lấy token đầu tiên có thể parse float
        for token in content.replace(",", ".").split():
            try:
                score = float(token)
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
    except Exception as e:
        print(f"[viquad_eval] Error parsing score: {e}")

    print(f"[viquad_eval] Could not parse valid score from judge output: {content}")
    return 0.0


def grade_viquad_answer_with_judge(
    context: str,
    question: str,
    gold_answers,
    pred_answer: str,
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
) -> float:
    """
    Xây prompt + gọi judge để chấm điểm 0..1.
    """
    prompt = build_judge_prompt(context, question, gold_answers, pred_answer)
    score = call_judge_api(
        prompt=prompt,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_model=judge_model,
    )
    return score


# =========================================================
# 4. FULL EVALUATION VỚI JUDGE
# =========================================================

def run_viquad_eval_with_judge(
    model,
    tokenizer,
    split: str = "validation",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 64,
    judge_base_url: str = "http://localhost:8000/v1",
    judge_api_key: str = "token-abc123",
    judge_model: str = "Qwen2.5-7B-Instruct",
    batch_size: int = 8,
    use_batching: bool = True,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    
    Evaluate UIT-ViQuAD2.0 với LLM-as-a-judge.

    - model, tokenizer: model ứng viên để sinh câu trả lời (vd. Qwen3-4B).
    - judge_*: config cho Qwen2.5-7B deploy qua vLLM.
    - batch_size: Number of samples to process in parallel (if use_batching=True)
    - use_batching: If True, generate answers in batches for faster inference

    Trả về:
        metrics = {"overall_judge_score": avg_score}
        counts = {"overall": n_total}
    """
    ds = load_viquad_split(split)

    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Filter out items without answers
    valid_items = [
        item for item in ds 
        if item.get("answers") and item["answers"].get("text")
    ]

    if not valid_items:
        return {"overall_judge_score": math.nan}, {"overall": 0}

    contexts = [item["context"] for item in valid_items]
    questions = [item["question"] for item in valid_items]
    answers_list = [item["answers"] for item in valid_items]

    # Generate answers (batched or sequential)
    if use_batching and len(valid_items) > 1:
        pred_answers = generate_viquad_answers_batched(
            model,
            tokenizer,
            contexts=contexts,
            questions=questions,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
    else:
        # Sequential generation (original method)
        pred_answers = []
        for context, question in zip(contexts, questions):
            answer = generate_viquad_answer(
                model,
                tokenizer,
                context=context,
                question=question,
                max_new_tokens=max_new_tokens,
            )
            pred_answers.append(answer)

    # Grade answers with judge
    n_total = 0
    score_sum = 0.0

    for context, question, gold_answers, pred_answer in tqdm(
        zip(contexts, questions, answers_list, pred_answers),
        desc=f"[ViQuAD2-Judge] split={split}",
        total=len(valid_items),
        leave=False,
    ):
        score = grade_viquad_answer_with_judge(
            context=context,
            question=question,
            gold_answers=gold_answers,
            pred_answer=pred_answer,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            judge_model=judge_model,
        )

        score_sum += score
        n_total += 1

    if n_total > 0:
        avg_score = score_sum / n_total
    else:
        avg_score = math.nan

    metrics = {
        "overall_judge_score": avg_score,
    }
    counts = {
        "overall": n_total,
    }
    return metrics, counts
