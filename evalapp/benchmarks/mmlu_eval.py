# mmlu_eval.py (phần liên quan đến MMLU eval)

import math
import torch
from tqdm import tqdm
from datasets import load_dataset
from typing import Dict, List, Optional, Tuple


CHOICE_LETTERS = ["A", "B", "C", "D"]

DEFAULT_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 
    'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
    'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 
    'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 
    'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 
    'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 
    'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 
    'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 
    'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 
    'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 
    'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 
    'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

# Cache dataset theo subject + split để không load lại nhiều lần
_MMLU_SUBJECT_CACHE: Dict[Tuple[str, str], "datasets.Dataset"] = {}


def load_mmlu_subject(subject: str, split: str = "test"):
    """
    Load một subject MMLU: load_dataset('cais/mmlu', subject)[split]
    và cache lại để dùng nhiều lần.
    """
    key = (subject, split)
    if key in _MMLU_SUBJECT_CACHE:
        return _MMLU_SUBJECT_CACHE[key]

    print(f"[mmlu_eval] Loading cais/mmlu subject='{subject}', split='{split}' ...")
    ds = load_dataset("cais/mmlu", subject)[split]
    _MMLU_SUBJECT_CACHE[key] = ds
    return ds


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    lines = [question.strip()]
    for i, c in enumerate(choices):
        lines.append(f"{CHOICE_LETTERS[i]}. {c}")
    lines.append("\nAnswer:")
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

    ans_tensor = torch.tensor(ans_ids, dtype=torch.long, device=device)  # [ans_len]
    ans_tensor = ans_tensor.view(1, ans_len, 1)  # [1, ans_len, 1]

    token_logprobs = log_probs.gather(-1, ans_tensor).squeeze(-1)  # [1, ans_len]
    token_logprobs = token_logprobs.sum(dim=1)  # [1]
    return float(token_logprobs.item())


@torch.no_grad()
def compute_logprobs_for_all_choices_batched(
    model,
    tokenizer,
    prompt: str,
    choices: List[str],
    max_tokens: int = 512,
) -> List[float]:
    """
    Tính log P(choice_letter | prompt) cho TẤT CẢ choices trong một forward pass.
    Đây là optimization quan trọng để tăng tốc inference.
    
    Returns: list of logprobs [logprob_A, logprob_B, logprob_C, logprob_D]
    """
    device = next(model.parameters()).device
    
    # Tokenize prompt once
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens - 10,  # Reserve space for answer tokens
    )
    prompt_ids = prompt_enc["input_ids"].to(device)  # [1, prompt_len]
    prompt_len = prompt_ids.size(1)
    
    # Tokenize all answer letters
    choice_letters = CHOICE_LETTERS[:len(choices)]
    choice_token_ids = []
    choice_lengths = []
    
    for letter in choice_letters:
        ans_ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
        choice_token_ids.append(ans_ids)
        choice_lengths.append(len(ans_ids))
    
    # Create batched input: [prompt + " " + choice_A, prompt + " " + choice_B, ...]
    batched_texts = [prompt + " " + letter for letter in choice_letters]
    
    # Tokenize all together with padding
    batched_enc = tokenizer(
        batched_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    batched_input_ids = batched_enc["input_ids"].to(device)  # [num_choices, seq_len]
    attention_mask = batched_enc["attention_mask"].to(device)
    
    # Forward pass for all choices at once
    outputs = model(batched_input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [num_choices, seq_len, vocab_size]
    
    # Extract logprobs for each choice
    scores = []
    for i, letter in enumerate(choice_letters):
        seq_len = batched_input_ids[i].size(0)
        ans_len = choice_lengths[i]
        
        # Find where answer tokens start (after prompt + space)
        # The answer tokens are at the end
        start = seq_len - ans_len
        if start < 0:
            scores.append(float("-inf"))
            continue
            
        # Extract logits for answer tokens
        logits_ans = logits[i, start - 1 : seq_len - 1, :]  # [ans_len, vocab_size]
        log_probs = torch.log_softmax(logits_ans, dim=-1)
        
        # Gather logprobs for actual answer tokens
        ans_tensor = torch.tensor(choice_token_ids[i], dtype=torch.long, device=device)
        ans_tensor = ans_tensor.view(ans_len, 1)  # [ans_len, 1]
        
        token_logprobs = log_probs.gather(-1, ans_tensor).squeeze(-1)  # [ans_len]
        total_logprob = token_logprobs.sum().item()
        scores.append(total_logprob)
    
    return scores


def predict_mmlu_single(
    model,
    tokenizer,
    question: str,
    choices: List[str],
    use_batching: bool = True,
) -> int:
    """
    Trả về index (0..3) của đáp án được chọn.
    
    Args:
        use_batching: If True, compute all choices in one forward pass (faster).
    """
    prompt = format_mmlu_prompt(question, choices)
    
    if use_batching:
        # Optimized: compute all choices in one forward pass
        scores = compute_logprobs_for_all_choices_batched(model, tokenizer, prompt, choices)
    else:
        # Original: compute each choice separately
        scores = []
        for i in range(len(choices)):
            letter = CHOICE_LETTERS[i]
            lp = compute_logprob_for_choice(model, tokenizer, prompt, letter)
            scores.append(lp)
    
    scores_t = torch.tensor(scores)
    return int(scores_t.argmax().item())


def run_mmlu_eval(
    model,
    tokenizer,
    subjects: Optional[List[str]] = None,
    split: str = "test",
    max_samples_per_subject: Optional[int] = 20,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Evaluate MMLU theo từng subject.

    Thay vì load 'all', ta load:
        load_dataset("cais/mmlu", subject)[split]
    cho từng subject.

    Trả về:
        - acc_by_subject: {subject: acc, "overall": acc_overall}
        - count_by_subject: {subject: n, "overall": total_n}
    """
    if subjects is None or len(subjects) == 0:
        subjects = DEFAULT_SUBJECTS

    acc_by_subject: Dict[str, float] = {}
    count_by_subject: Dict[str, int] = {}

    total_correct = 0
    total_count = 0

    for subject in tqdm(subjects, desc="[MMLU] Evaluating subjects"):
        ds = load_mmlu_subject(subject, split=split)

        if max_samples_per_subject is not None and max_samples_per_subject > 0:
            ds = ds.select(range(min(max_samples_per_subject, len(ds))))

        if len(ds) == 0:
            continue

        correct = 0

        for item in tqdm(ds, desc=f"[MMLU] {subject}", leave=False):
            question = item["question"]
            choices = item["choices"]
            gold_idx = int(item["answer"])  # index 0..3

            pred_idx = predict_mmlu_single(model, tokenizer, question, choices)
            if pred_idx == gold_idx:
                correct += 1

        n = len(ds)
        acc = correct / n
        acc_by_subject[subject] = acc
        count_by_subject[subject] = n

        total_correct += correct
        total_count += n

    if total_count > 0:
        acc_by_subject["overall"] = total_correct / total_count
        count_by_subject["overall"] = total_count
    else:
        acc_by_subject["overall"] = math.nan
        count_by_subject["overall"] = 0

    return acc_by_subject, count_by_subject

# def parse_subjects_cfg(cfg_text: str) -> List[str]:
#     """
#     Mỗi dòng trong textbox config:
#         subject_name
#         subject_name: 50
#     -> lấy phần trước dấu ':' làm subject.
#     """
#     subjects = []
#     for line in cfg_text.splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         if ":" in line:
#             subj = line.split(":", 1)[0].strip()
#         else:
#             subj = line
#         subjects.append(subj)
#     return subjects