
import math
import string
from typing import Dict, Tuple, Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


def _normalize_text(s: str) -> str:
    """
    Chuẩn hoá đơn giản: lowercase + strip punctuation + trim spaces.
    Dùng cho so sánh target vs prediction.
    """
    s = s.strip()
    # bỏ punctuation ở 2 đầu
    s = s.strip(string.punctuation + "“”‘’\"'、。、…")
    s = " ".join(s.split())
    return s.lower()


def run_lambada_vi_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    split: str = "test",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 10,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Evaluate model trên VLSP-2023 LAMBADA_vi benchmark.

    Dataset format (ví dụ):
        ds_lambada_vi = load_dataset("vlsp-2023-vllm/lambada_vi")
        ds_lambada_vi['test'][i]:
        {
            'text': ...  # context + target_word
            'context': '...',
            'target_word': 'Herøy' / 'vương quốc' / ...
            'metadata': {...}
        }

    Ta dùng:
        - input:  example["context"]
        - target: example["target_word"]
    Model phải sinh tiếp đoạn text, và ta check xem
    token(s) đầu tiên trong phần sinh ra có khớp target_word hay không (sau normalize).

    Return:
        acc_by_subject = {"overall": accuracy}
        count_by_subject = {"overall": num_examples}
    """
    device = next(model.parameters()).device
    ds = load_dataset("vlsp-2023-vllm/lambada_vi", split=split)

    total = 0
    correct = 0

    for i, ex in enumerate(ds):
        if max_samples is not None and total >= max_samples:
            break

        context = ex.get("context", "")
        target_word = ex.get("target_word", "")
        if not context or not target_word:
            continue

        prompt = context.rstrip()
        # Tokenize + move to same device
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # greedy decoding, không sampling
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Tách phần model sinh thêm sau context
        if full_text.startswith(prompt):
            suffix = full_text[len(prompt):]
        else:
            # trong trường hợp tokenization hơi lệch, fallback: lấy luôn full_text
            suffix = full_text

        suffix = suffix.strip()

        # Target có thể là 1 hoặc nhiều từ ("Herøy" vs "vương quốc")
        target_tokens = target_word.split()
        pred_tokens = suffix.split()

        if len(pred_tokens) == 0:
            pred_span = ""
        else:
            # lấy số token bằng với số token của target
            pred_span = " ".join(pred_tokens[: len(target_tokens)])

        gold_norm = _normalize_text(target_word)
        pred_norm = _normalize_text(pred_span)

        is_correct = int(gold_norm == pred_norm)

        total += 1
        correct += is_correct

    acc = correct / total if total > 0 else 0.0

    acc_by_subject = {"overall": acc}
    count_by_subject = {"overall": total}
    return acc_by_subject, count_by_subject