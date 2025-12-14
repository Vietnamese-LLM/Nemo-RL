# vietnews_summarization_eval.py
"""
Vietnamese News Summarization Evaluation (LLM-as-a-Judge)
---------------------------------------------------------

Dataset: nam194/vietnews

Example entry:
{
 'guid': 2,
 'title': 'Án chung_thân cho đối_tượng giết vợ vì ghen_tuông',
 'abstract': 'Ngày 19/9 ...',
 'article': 'Theo cáo_trạng ...'
}

YÊU CẦU:
- Data đang bị preprocess "_" để nối từ: cần convert "_" → " " trước khi dùng.
- Candidate model: generate summary từ article.
- Judge model (vLLM): đánh giá similarity 0..1 giữa generated summary và ground truth abstract.
"""

import math
from typing import Dict, Tuple, Optional
import requests
import torch
from datasets import load_dataset
from tqdm.auto import tqdm


# =========================================================
# 1. LOAD DATA
# =========================================================

_VIETNEWS_CACHE = {}  # cache theo split


def clean_underscores(text: str) -> str:
    """Chuyển toàn bộ "_" -> " "."""
    if not isinstance(text, str):
        return text
    return text.replace("_", " ").strip()


def load_vietnews_split(split: str = "validation"):
    """
    Load dataset và replace "_" trong title / abstract / article.
    Cache lại để dùng nhiều lần.
    """
    global _VIETNEWS_CACHE
    if split in _VIETNEWS_CACHE:
        return _VIETNEWS_CACHE[split]

    print(f"[vietnews_eval] Loading nam194/vietnews split='{split}' ...")
    ds = load_dataset("nam194/vietnews")[split]

    # làm sạch "_" -> " "
    cleaned = []
    for ex in ds:
        cleaned.append({
            "guid": ex["guid"],
            "title": clean_underscores(ex["title"]),
            "abstract": clean_underscores(ex["abstract"]),
            "article": clean_underscores(ex["article"]),
        })
    _VIETNEWS_CACHE[split] = cleaned
    return cleaned


# =========================================================
# 2. PROMPT CHO TÓM TẮT
# =========================================================

def format_vietnews_prompt(article: str) -> str:
    """
    Prompt sinh summary bằng tiếng Việt.
    Bạn có thể tuỳ chỉnh sâu hơn nếu cần.
    """
    prompt = (
        "Hãy tóm tắt ngắn gọn nội dung bài báo sau bằng tiếng Việt:\n\n"
        f"{article.strip()}\n\n"
        "Tóm tắt:"
    )
    return prompt


@torch.no_grad()
def generate_vietnews_summary(
    model,
    tokenizer,
    article: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Sinh summary từ candidate model (Qwen3-4B / Qwen2.5 / …).
    """
    device = next(model.parameters()).device
    prompt = format_vietnews_prompt(article)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    input_ids = enc["input_ids"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )

    gen_ids = output_ids[0, input_ids.size(1):]
    ans = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return ans.strip()


# =========================================================
# 3. LLM-AS-A-JUDGE (Qwen2.5-7B)
# =========================================================

def build_judge_prompt(article: str, pred_summary: str, gold_summary: str) -> str:
    """
    Prompt cho judge model đánh giá similarity 0..1 giữa summary và gold.
    """
    prompt = f"""Bạn là trợ lý đánh giá chất lượng tóm tắt tiếng Việt.

Nhiệm vụ:
- So sánh bản tóm tắt do mô hình sinh ra với bản tóm tắt tham chiếu.
- Cho điểm từ 0 đến 1:
  * 1.0: đúng nghĩa, đầy đủ ý chính, ngắn gọn tốt.
  * 0.5: đúng một phần, còn thiếu/sai ý.
  * 0.0: sai hoàn toàn / không liên quan.
- Chỉ in ra MỘT SỐ thực duy nhất trong khoảng [0, 1].
- Không giải thích thêm.

Bài báo:
{article}

Tóm tắt tham chiếu:
{gold_summary}

Tóm tắt mô hình:
{pred_summary}

Điểm (0..1):
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
    Gọi judge model qua OpenAI-compatible API (vLLM).
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
        print(f"[vietnews_eval] Judge API error: {e}")
        return 0.0

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vietnews_eval] Failed to parse judge JSON: {e}")
        return 0.0

    # parse số
    try:
        for tok in content.replace(",", ".").split():
            try:
                score = float(tok)
                if 0 <= score <= 1:
                    return score
            except:
                pass
    except:
        pass

    print(f"[vietnews_eval] Invalid judge output: {content}")
    return 0.0


def grade_vietnews_summary_with_judge(
    article: str,
    pred_summary: str,
    gold_summary: str,
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
) -> float:
    """
    Gửi prompt đến judge model => trả về score 0..1.
    """
    prompt = build_judge_prompt(article, pred_summary, gold_summary)
    return call_judge_api(
        prompt=prompt,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_model=judge_model,
    )


# =========================================================
# 4. FULL EVALUATION
# =========================================================

def run_vietnews_summarization_eval(
    model,
    tokenizer,
    split: str = "validation",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 128,
    judge_base_url: str = "http://localhost:8000/v1",
    judge_api_key: str = "token-abc123",
    judge_model: str = "Qwen2.5-7B-Instruct",
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Evaluate summarization model using LLM-as-a-judge on VietNews.

    Trả về:
        metrics = {"overall_judge_score": avg_score}
        counts = {"overall": n}
    """
    ds = load_vietnews_split(split)

    if max_samples is not None and max_samples > 0:
        ds = ds[:max_samples]

    n_total = 0
    score_sum = 0.0

    for item in tqdm(ds, desc=f"[VietNews-Judge] split={split}", leave=False):
        article = item["article"]
        gold_summary = item["abstract"]

        pred_summary = generate_vietnews_summary(
            model,
            tokenizer,
            article=article,
            max_new_tokens=max_new_tokens,
        )

        score = grade_vietnews_summary_with_judge(
            article=article,
            pred_summary=pred_summary,
            gold_summary=gold_summary,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            judge_model=judge_model,
        )

        score_sum += score
        n_total += 1

    avg_score = score_sum / n_total if n_total > 0 else math.nan

    metrics = {
        "overall_judge_score": avg_score,
    }
    counts = {
        "overall": n_total,
    }
    return metrics, counts
