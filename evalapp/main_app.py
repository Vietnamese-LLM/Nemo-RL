# app.py
"""
LLM Model Evaluation Explorer

- Chọn model family (Qwen3-4B / Qwen3-4B-Instruct / ...)
- Chọn checkpoint folder trong weights/<model_family>/ (vd. base, exp1, ...)
- Load model từ checkpoint
- Chạy evaluation trên nhiều benchmark:
    * MMLU
    * ZaloMath
    * ViQuAD (open-ended QA, judge bằng Qwen2.5-7B via vLLM)
    * VietNews Summarization (judge bằng Qwen2.5-7B via vLLM)
    * VNHSGE-V (MCQ)
    * VMLU (MCQ)
"""

import math
import gradio as gr
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt  # (not used yet, but keep for future training curve)

from model_utils import (
    MODEL_REGISTRY,
    load_qwen_model,
    get_current_model_and_tokenizer,
    list_checkpoints_in_folder,  # not used yet, but keep for future training curve
)
from benchmarks.mmlu_eval import DEFAULT_SUBJECTS, run_mmlu_eval

# Other benchmarks (ensure path/module is correct with your repo)
from benchmarks.vmlu_eval import run_vmlu_eval
from benchmarks.vnhsge_eval import run_vnhsge_eval
from benchmarks.zaloai_math_eval import run_zalo_math_eval
from benchmarks.viquad_eval import run_viquad_eval_with_judge
from benchmarks.vietnews_summarization_eval import run_vietnews_summarization_eval


# ================== Config ==================

# Root of Checkpoint Folders
WEIGHTS_ROOT = Path("/home/mingzhe/Nemo-RL/results/")

# Maximum number of samples (0 = full) for MMLU / Zalo / VMLU / VNHSGE...
MAX_SAMPLES = 0

# ================== Helpers: scan checkpoint folders ==================

def list_weight_folders_for_model(model_key: str):
    """
    Return list of subfolders in results/<model_key>/ (only directories).
    Example: ['base', 'run_1', ...]
    """
    base = WEIGHTS_ROOT / model_key
    if not base.exists() or not base.is_dir():
        return []
    folders = [p.name for p in base.iterdir() if p.is_dir()]
    return sorted(folders)


def ui_update_checkpoint_list(model_key: str):
    """
    When user changes model family → refresh list of checkpoint folders.
    Also set the first path to the textbox.
    """
    folders = list_weight_folders_for_model(model_key)
    if not folders:
        return gr.update(choices=[], value=None), ""
    first = folders[0]
    full_path = str(WEIGHTS_ROOT / model_key / first)
    return gr.update(choices=folders, value=first), full_path


def ui_update_ckpt_path(model_key: str, ckpt_folder: str):
    """
    When user selects a folder in the checkpoint dropdown → update path textbox.
    """
    if not ckpt_folder:
        return ""
    return str(WEIGHTS_ROOT / model_key / ckpt_folder)


# ================== UI callbacks: load model & benchmarks ==================

def ui_load_model(model_key: str, ckpt_folder: str):
    print("Loading model")

    checkpoint_path = None
    if ckpt_folder:
        checkpoint_path = str(WEIGHTS_ROOT / model_key / ckpt_folder)

    try:
        model, tokenizer, path = load_qwen_model(model_key, checkpoint_path=checkpoint_path)
        msg = f"✅ Loaded **{model_key}** from `{path}`"

        enable = gr.update(interactive=True)
        return (
            msg,
            path,   # update textbox path
            enable, # MMLU
            enable, # ZaloMath
            enable, # ViQuAD
            enable, # VietNews
            enable, # VNHSGE-V
            enable, # VMLU
        )
    except Exception as e:
        disable = gr.update(interactive=False)
        return (
            f"❌ Error loading model: {e}",
            "",
            disable,
            disable,
            disable,
            disable,
            disable,
            disable,
        )


def ui_run_single_mmlu():
    print("Running MMLU")
    model, tokenizer = get_current_model_and_tokenizer()
    if model is None or tokenizer is None:
        return "❌ Please load a model first."

    subjects = DEFAULT_SUBJECTS
    max_samples = MAX_SAMPLES

    acc_by_subject, count_by_subject = run_mmlu_eval(
        model,
        tokenizer,
        subjects=subjects,
        split="test",
        max_samples_per_subject=max_samples if max_samples > 0 else None,
    )

    lines = []
    lines.append("### MMLU Single-Checkpoint Results\n")
    for subj, acc in tqdm(acc_by_subject.items()):
        if subj == "overall":
            continue
        n = count_by_subject.get(subj, 0)
        lines.append(f"- **{subj}**: {acc:.4f} (n={n})")
    lines.append("")
    lines.append(
        f"**Overall**: {acc_by_subject['overall']:.4f} (total n={count_by_subject['overall']})"
    )
    return "\n".join(lines)


def ui_run_single_zalo():
    print("Running ZaloMath")
    model, tokenizer = get_current_model_and_tokenizer()
    if model is None or tokenizer is None:
        return "❌ Please load a model first."

    acc_by_subject, count_by_subject = run_zalo_math_eval(
        model,
        tokenizer,
        json_path="./Elementary-Math-Solving-Zalo-AI-2023/datasets/math_test_with_hand_label.json",
        max_samples=None,
    )
    print("Done")

    lines = []
    lines.append("### ZaloMath Single-Checkpoint Results\n")
    for subj, acc in acc_by_subject.items():
        if subj == "overall":
            continue
        n = count_by_subject.get(subj, 0)
        lines.append(f"- **{subj}**: {acc:.4f} (n={n})")
    lines.append("")
    lines.append(
        f"**Overall**: {acc_by_subject['overall']:.4f} (total n={count_by_subject['overall']})"
    )
    return "\n".join(lines)


def ui_run_single_viquad():
    print("Running ViQuAD (judge)")
    model, tokenizer = get_current_model_and_tokenizer()
    if model is None or tokenizer is None:
        return "❌ Please load a model first."

    metrics, counts = run_viquad_eval_with_judge(
        model,
        tokenizer,
        split="validation",
        max_samples=None,           # hoặc số nhỏ để test nhanh
        max_new_tokens=64,
        judge_base_url="http://localhost:8000/v1",
        judge_api_key="token-abc123",
        judge_model="Qwen/Qwen2.5-7B-Instruct",
    )

    score = metrics.get("overall_judge_score", float("nan"))
    n = counts.get("overall", 0)

    lines = []
    lines.append("### ViQuAD Single-Checkpoint Results (LLM-as-a-Judge)\n")
    lines.append(f"- **Judge score**: {score:.4f}")
    lines.append(f"- **Number of questions**: {n}")
    return "\n".join(lines)


def ui_run_single_vietnews():
    print("Running VietNews summarization (judge)")
    model, tokenizer = get_current_model_and_tokenizer()
    if model is None or tokenizer is None:
        return "❌ Please load a model first."

    metrics, counts = run_vietnews_summarization_eval(
        model,
        tokenizer,
        split="validation",
        max_samples=None,
        max_new_tokens=128,
        judge_base_url="http://localhost:8000/v1",
        judge_api_key="token-abc123",
        judge_model="Qwen/Qwen2.5-7B-Instruct",
    )

    score = metrics.get("overall_judge_score", float("nan"))
    n = counts.get("overall", 0)

    lines = []
    lines.append("### VietNews Summarization Single-Checkpoint Results (LLM-as-a-Judge)\n")
    lines.append(f"- **Judge score**: {score:.4f}")
    lines.append(f"- **Number of articles**: {n}")
    return "\n".join(lines)


def ui_run_single_vnhsge():
    print("Running VNHSGE-V")
    model, tokenizer = get_current_model_and_tokenizer()
    if model is None or tokenizer is None:
        return "❌ Please load a model first."

    results = run_vnhsge_eval(
        model,
        tokenizer,
        root_dir="./VNHSGE-V",
        max_samples_per_subset=None,
    )

    per_subset = results["per_subject_subset"]
    per_overall = results["per_subject_overall"]
    overall = results["overall"]

    lines = []
    lines.append("### VNHSGE-V Single-Checkpoint Results\n")
    lines.append(f"**Overall**: {overall['accuracy']:.4f} (N={overall['num_questions']})\n")

    for subject, s_res in per_overall.items():
        lines.append(f"#### {subject}")
        lines.append(
            f"- Subject overall: {s_res['accuracy']:.4f} "
            f"(N={s_res['num_questions']})"
        )
        subset_dict = per_subset.get(subject, {})
        for subset_name, subinfo in subset_dict.items():
            lines.append(
                f"  - {subset_name}: {subinfo['accuracy']:.4f} "
                f"(N={subinfo['num_questions']})"
            )
        lines.append("")

    return "\n".join(lines)


def ui_run_single_vmlu():
    print("Running VMLU")
    model, tokenizer = get_current_model_and_tokenizer()
    if model is None or tokenizer is None:
        return "❌ Please load a model first."

    acc_by_subject, count_by_subject = run_vmlu_eval(
        model,
        tokenizer,
        jsonl_path="./VMLU/valid.jsonl",
        max_samples=None,
    )

    acc = acc_by_subject.get("overall", float("nan"))
    n = count_by_subject.get("overall", 0)

    lines = []
    lines.append("### VMLU Single-Checkpoint Results\n")
    lines.append(f"- **Accuracy**: {acc:.4f}")
    lines.append(f"- **Number of questions**: {n}")
    return "\n".join(lines)


def ui_refresh_models():
    keys = list(MODEL_REGISTRY.keys())
    return gr.update(choices=keys, value=keys[0] if keys else None)


def run_all_benchmarks_on_checkpoint(model_key, ckpt_path):
    """
    Load model from checkpoint_path and run all benchmarks:
      - MMLU
      - ZaloMath
      - ViQuAD (judge)
      - VietNews (judge)
      - VNHSGE-V
      - VMLU
    Return:
      {
         "MMLU": overall_score,
         "ZaloMath": ...,
         ...
      }
    """
    model, tokenizer, _ = load_qwen_model(model_key, checkpoint_path=ckpt_path)

    out = {}

    # ---- MMLU ----
    acc_by_subject, count_by_subject = run_mmlu_eval(
        model, tokenizer,
        subjects=DEFAULT_SUBJECTS,
        split="test",
        max_samples_per_subject=3
    )
    out["MMLU"] = acc_by_subject["overall"]

    # ---- ZaloMath ----
    acc_by_subject, count_by_subject = run_zalo_math_eval(
        model, tokenizer,
        max_samples=3,
        json_path="./Elementary-Math-Solving-Zalo-AI-2023/datasets/math_test.json"
    )
    out["ZaloMath"] = acc_by_subject["overall"]

    # ---- ViQuAD ----
    metrics, counts = run_viquad_eval_with_judge(
        model, tokenizer,
        split="validation",
        max_samples=3,
        judge_base_url="http://localhost:8000/v1",
        judge_api_key="token-abc123",
        judge_model="Qwen/Qwen2.5-7B-Instruct",
    )
    out["ViQuAD"] = metrics["overall_judge_score"]

    # ---- VietNews ----
    metrics, counts = run_vietnews_summarization_eval(
        model, tokenizer,
        split="validation",
        max_samples=3,
        judge_base_url="http://localhost:8000/v1",
        judge_api_key="token-abc123",
        judge_model="Qwen/Qwen2.5-7B-Instruct",
    )
    out["VietNews"] = metrics["overall_judge_score"]

    # ---- VNHSGE-V ----
    results = run_vnhsge_eval(model, tokenizer, max_samples_per_subset=3, root_dir="./VNHSGE-V")
    out["VNHSGE-V"] = results["overall"]["accuracy"]

    # ---- VMLU ----
    acc_by_subject, count_by_subject = run_vmlu_eval(
        model, tokenizer,
        jsonl_path="./VMLU/valid.jsonl",
        max_samples=3,
    )
    out["VMLU"] = acc_by_subject["overall"]

    return out



# ================== Tab 2: select family + list step_xxx ==================

def ui_list_ckpts_for_model_curve(model_key: str):
    """
    Tab 2:
      - Receive model family
      - Liệt kê tất cả checkpoints trong weights/<model_key>/
        match pattern step_xxx / step-xxxxx / step_000500 ...
    """
    folder = WEIGHTS_ROOT / model_key
    if not folder.exists() or not folder.is_dir():
        return f"❌ Folder not found: {folder}"

    try:
        ckpts = list_checkpoints_in_folder(str(folder), pattern=r"step[_-]?(\d+)")
    except Exception as e:
        return f"❌ Error scanning {folder}: {e}"

    if not ckpts:
        return f"⚠️ No checkpoints matching 'step*' found in {folder}"

    lines = []
    lines.append(f"Checkpoints in `{folder}`:\n")
    for step, path in ckpts:
        lines.append(f"- step {step:08d}: {path}")
    return "\n".join(lines)




# def ui_analyze_training_curve(model_key: str):
    """
    Tab 2: Analyze training curve for 1 model family:
        - Scan weights/<model_key>/ to find step_xxx
        - For each checkpoint:
              load_qwen_model(model_key, checkpoint_path=ckpt_path)
              run_mmlu_eval -> get overall accuracy
              run_zalo_math_eval -> get overall accuracy
              run_viquad_eval_with_judge -> get overall judge score
              run_vietnews_summarization_eval -> get overall judge score
              run_vnhsge_eval -> get overall accuracy
              run_vmlu_eval -> get overall accuracy
        - Plot accuracy vs step for each benchmark
    """
    folder = WEIGHTS_ROOT / model_key
    if not folder.exists() or not folder.is_dir():
        return None, f"❌ Folder not found: {folder}"

    try:
        ckpts = list_checkpoints_in_folder(str(folder), pattern=r"step[_-]?(\d+)")
    except Exception as e:
        return None, f"❌ Error scanning {folder}: {e}"

    if not ckpts:
        return None, f"⚠️ No checkpoints matching 'step*' found in {folder}"

    steps = []
    accuracies = []

    print(f"[analyze_curve] Found {len(ckpts)} checkpoints in {folder}")
    for step, ckpt_path in ckpts:
        print(f"[analyze_curve] Evaluating step {step} at {ckpt_path}")
        model, tokenizer, _ = load_qwen_model(model_key, checkpoint_path=ckpt_path)

        acc_by_subject, count_by_subject = run_mmlu_eval(
            model,
            tokenizer,
            subjects=DEFAULT_SUBJECTS,
            split="test",
            max_samples_per_subject=3,
        )
        overall = acc_by_subject.get("overall", float("nan"))
        steps.append(step)
        accuracies.append(overall)

    # Plot curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, accuracies, marker="o")  # không set màu để tuân thủ rule
    ax.set_xlabel("Training step")
    ax.set_ylabel("MMLU overall accuracy")
    ax.set_title(f"MMLU Accuracy vs Training Step ({model_key})")
    ax.grid(True, alpha=0.3)

    # Summary text
    if steps:
        best_idx = max(range(len(steps)), key=lambda i: accuracies[i])
        best_step = steps[best_idx]
        best_acc = accuracies[best_idx]
        first_step = steps[0]
        last_step = steps[-1]
    else:
        best_step, best_acc, first_step, last_step = "-", float("nan"), "-", "-"

    lines = []
    lines.append("### Training Curve Summary\n")
    lines.append(f"- Model family: **{model_key}**")
    lines.append(f"- Number of checkpoints: **{len(steps)}**")
    lines.append(f"- Best step: **{best_step}**  (overall acc = {best_acc:.4f})")
    if steps:
        lines.append(f"- First step: {first_step}, last step: {last_step}")
    return fig, "\n".join(lines)

def ui_analyze_training_curve(model_key: str):
    """
    Tab 2: Analyze training curve cho 1 model family:
      - Quét weights/<model_key>/ tìm step_xxx
      - Cho từng checkpoint:
          load_qwen_model(model_key, checkpoint_path=ckpt_path)
          Chạy TẤT CẢ benchmark:
            * MMLU (overall acc)
            * ZaloMath (overall acc)
            * ViQuAD (overall judge score)
            * VietNews (overall judge score)
            * VNHSGE-V (overall acc)
            * VMLU (overall acc)
      - Vẽ nhiều plot, mỗi benchmark 1 plot
      - Summary: best step cho từng benchmark
    """
    folder = WEIGHTS_ROOT / model_key
    if not folder.exists() or not folder.is_dir():
        return (None, None, None, None, None, None,
                f"❌ Folder not found: {folder}")

    try:
        ckpts = list_checkpoints_in_folder(str(folder), pattern=r"step[_-]?(\d+)")
    except Exception as e:
        return (None, None, None, None, None, None,
                f"❌ Error scanning {folder}: {e}")

    if not ckpts:
        return (None, None, None, None, None, None,
                f"⚠️ No checkpoints matching 'step*' found in {folder}")

    steps = []
    mmlu_scores = []
    zalo_scores = []
    viq_scores = []
    vietnews_scores = []
    vnh_scores = []
    vmlu_scores = []

    print(f"[analyze_curve] Found {len(ckpts)} checkpoints in {folder}")
    for step, ckpt_path in ckpts:
        print(f"[analyze_curve] Evaluating ALL benchmarks at step {step} ({ckpt_path})")
        model, tokenizer, _ = load_qwen_model(model_key, checkpoint_path=ckpt_path)

        # ---- MMLU ----
        acc_mmlu, cnt_mmlu = run_mmlu_eval(
            model,
            tokenizer,
            subjects=DEFAULT_SUBJECTS,
            split="test",
            max_samples_per_subject=3,
        )
        mmlu_overall = acc_mmlu.get("overall", float("nan"))

        # ---- ZaloMath ----
        acc_zalo, cnt_zalo = run_zalo_math_eval(
            model,
            tokenizer,
            json_path="./Elementary-Math-Solving-Zalo-AI-2023/datasets/math_test.json",
            max_samples=3,
        )
        zalo_overall = acc_zalo.get("overall", float("nan"))

        # ---- ViQuAD (judge) ----
        metrics_viq, counts_viq = run_viquad_eval_with_judge(
            model,
            tokenizer,
            split="validation",
            max_samples=3,
            max_new_tokens=64,
            judge_base_url="http://localhost:8000/v1",
            judge_api_key="token-abc123",
            judge_model="Qwen/Qwen2.5-7B-Instruct",
        )
        viq_overall = metrics_viq.get("overall_judge_score", float("nan"))

        # ---- VietNews (judge) ----
        metrics_vn, counts_vn = run_vietnews_summarization_eval(
            model,
            tokenizer,
            split="validation",
            max_samples=3,
            max_new_tokens=128,
            judge_base_url="http://localhost:8000/v1",
            judge_api_key="token-abc123",
            judge_model="Qwen/Qwen2.5-7B-Instruct",
        )
        vietnews_overall = metrics_vn.get("overall_judge_score", float("nan"))

        # ---- VNHSGE-V ----
        results_vnh = run_vnhsge_eval(
            model,
            tokenizer,
            root_dir="./VNHSGE-V",
            max_samples_per_subset=3,
        )
        overall_vnh = results_vnh["overall"]
        vnh_overall = overall_vnh.get("accuracy", float("nan"))

        # ---- VMLU ----
        acc_vmlu, cnt_vmlu = run_vmlu_eval(
            model,
            tokenizer,
            jsonl_path="./VMLU/valid.jsonl",
            max_samples=3,
        )
        vmlu_overall = acc_vmlu.get("overall", float("nan"))

        # push vào list
        steps.append(step)
        mmlu_scores.append(mmlu_overall)
        zalo_scores.append(zalo_overall)
        viq_scores.append(viq_overall)
        vietnews_scores.append(vietnews_overall)
        vnh_scores.append(vnh_overall)
        vmlu_scores.append(vmlu_overall)

    # Helper to create plot for each benchmark
    def make_curve_fig(title, steps, scores):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(steps, scores, marker="o")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return fig

    fig_mmlu = make_curve_fig("MMLU vs Training Step", steps, mmlu_scores)
    fig_zalo = make_curve_fig("ZaloMath vs Training Step", steps, zalo_scores)
    fig_viq = make_curve_fig("ViQuAD (judge) vs Training Step", steps, viq_scores)
    fig_vn = make_curve_fig("VietNews (judge) vs Training Step", steps, vietnews_scores)
    fig_vnh = make_curve_fig("VNHSGE-V vs Training Step", steps, vnh_scores)
    fig_vmlu = make_curve_fig("VMLU vs Training Step", steps, vmlu_scores)

    # Helper tìm best step cho 1 list scores
    def best_step_for(scores):
        valid_idx = [
            i for i, s in enumerate(scores)
            if not (isinstance(s, float) and math.isnan(s))
        ]
        if not valid_idx:
            return None, float("nan")
        best_i = max(valid_idx, key=lambda i: scores[i])
        return steps[best_i], scores[best_i]

    mmlu_best_step, mmlu_best = best_step_for(mmlu_scores)
    zalo_best_step, zalo_best = best_step_for(zalo_scores)
    viq_best_step, viq_best = best_step_for(viq_scores)
    vn_best_step, vn_best = best_step_for(vietnews_scores)
    vnh_best_step, vnh_best = best_step_for(vnh_scores)
    vmlu_best_step, vmlu_best = best_step_for(vmlu_scores)

    lines = []
    lines.append("### Training Curve Summary (per benchmark)\n")
    lines.append(f"- Model family: **{model_key}**")
    lines.append(f"- Number of checkpoints: **{len(steps)}**")
    if steps:
        lines.append(f"- First step: {steps[0]}, last step: {steps[-1]}\n")

    def fmt_best(name, st, val):
        if st is None:
            return f"- {name}: no valid scores"
        return f"- {name}: best at step **{st}** (score = {val:.4f})"

    lines.append(fmt_best("MMLU", mmlu_best_step, mmlu_best))
    lines.append(fmt_best("ZaloMath", zalo_best_step, zalo_best))
    lines.append(fmt_best("ViQuAD (judge)", viq_best_step, viq_best))
    lines.append(fmt_best("VietNews (judge)", vn_best_step, vn_best))
    lines.append(fmt_best("VNHSGE-V", vnh_best_step, vnh_best))
    lines.append(fmt_best("VMLU", vmlu_best_step, vmlu_best))

    summary = "\n".join(lines)

    return fig_mmlu, fig_zalo, fig_viq, fig_vn, fig_vnh, fig_vmlu, summary





# ================== Build Gradio app ==================

theme = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, css="""
#title-bar {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 4px;
}
#subtitle {
    opacity: 0.8;
    margin-bottom: 12px;
}
#analyze-btn > button {
    background-color: #f97316 !important;
    color: white !important;
    font-weight: 700;
    font-size: 15px;
    height: 44px;
}
""") as demo:

    gr.Markdown("<div id='title-bar'>LLM Model Evaluation Explorer</div>")
    gr.Markdown(
        "<div id='subtitle'>Evaluation for Qwen3-4B variants on multiple Vietnamese benchmarks</div>"
    )

    with gr.Tabs():
        # ===== TAB 1: Single checkpoint evaluation =====
        with gr.TabItem("📊 Single Checkpoint"):
            # chọn model family + checkpoint folder
            with gr.Row():
                with gr.Column(scale=1):
                    model_dd = gr.Dropdown(
                        label="Select model family",
                        choices=list(MODEL_REGISTRY.keys()),
                        value=list(MODEL_REGISTRY.keys())[0],
                    )
                    refresh_btn = gr.Button("🔄", size="sm")
                with gr.Column(scale=1):
                    ckpt_dd = gr.Dropdown(
                        label="Select checkpoint folder (weights/<model>/...)",
                        choices=[],
                        value=None,
                    )
                with gr.Column(scale=2):
                    starting_ckpt_tb = gr.Textbox(
                        label="Model / Starting checkpoint path",
                        placeholder="Will be set after selecting checkpoint",
                    )
       
            load_btn = gr.Button("Load model")
            load_status = gr.Markdown("Model not loaded yet.")

            # --- MMLU ---
            run_mmlu_btn = gr.Button("Run MMLU Evaluation", interactive=False)
            mmlu_results_tb = gr.Textbox(
                value="MMLU results will appear here.",
                label="MMLU results",
            )

            # --- ZaloMath ---
            run_zalo_btn = gr.Button("Run ZaloMath Evaluation", interactive=False)
            zalo_results_tb = gr.Textbox(
                value="ZaloMath results will appear here.",
                label="Zalo results",
            )

            # --- ViQuAD ---
            run_viquad_btn = gr.Button("Run ViQuAD Evaluation", interactive=False)
            viquad_results_tb = gr.Textbox(
                value="ViQuAD results will appear here.",
                label="ViQuAD results",
            )

            # --- VietNews summarization ---
            run_vietnews_btn = gr.Button("Run VietNews Summarization Evaluation", interactive=False)
            vietnews_results_tb = gr.Textbox(
                value="VietNews results will appear here.",
                label="VietNews results",
            )

            # --- VNHSGE-V ---
            run_vnhsge_btn = gr.Button("Run VNHSGE-V Evaluation", interactive=False)
            vnhsge_results_tb = gr.Textbox(
                value="VNHSGE-V results will appear here.",
                label="VNHSGE-V results",
            )

            # --- VMLU ---
            run_vmlu_btn = gr.Button("Run VMLU Evaluation", interactive=False)
            vmlu_results_tb = gr.Textbox(
                value="VMLU results will appear here.",
                label="VMLU results",
            )

        # ===== TAB 2: Training checkpoints browser =====
        # with gr.TabItem("📈 Training Checkpoints"):
        #     with gr.Row():
        #         with gr.Column(scale=1):
        #             model_curve_dd = gr.Dropdown(
        #                 label="Select model family",
        #                 choices=list(MODEL_REGISTRY.keys()),
        #                 value=list(MODEL_REGISTRY.keys())[0],
        #             )
        #         with gr.Column(scale=2):
        #             ckpt_list_tb = gr.Textbox(
        #                 label="Detected step_* checkpoints in weights/<model>/",
        #                 value="Select a model family to list checkpoints.",
        #                 lines=12,
        #             )
        #     analyze_btn = gr.Button("Analyze Training Curve", elem_id="analyze-btn")
        #     with gr.Row():
        #         with gr.Column(scale=2):
        #             curve_plot = gr.Plot(label="MMLU accuracy vs step")
        #         with gr.Column(scale=1):
        #             curve_summary = gr.Markdown(
        #                 value="Summary will appear here.",
        #                 label="Training curve summary",
        #             )

        with gr.TabItem("📈 Training Checkpoints"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_curve_dd = gr.Dropdown(
                        label="Select model family",
                        choices=list(MODEL_REGISTRY.keys()),
                        value=list(MODEL_REGISTRY.keys())[0],
                    )
                with gr.Column(scale=2):
                    ckpt_list_tb = gr.Textbox(
                        label="Detected step_* checkpoints in weights/<model>/",
                        value="Select a model family to list checkpoints.",
                        lines=12,
                    )

            analyze_btn = gr.Button("Analyze Training Curve (ALL benchmarks)", elem_id="analyze-btn")

            # 6 plots: mỗi benchmark 1 plot
            with gr.Row():
                with gr.Column():
                    mmlu_plot = gr.Plot(label="MMLU vs step")
                with gr.Column():
                    zalo_plot = gr.Plot(label="ZaloMath vs step")
            with gr.Row():
                with gr.Column():
                    viq_plot = gr.Plot(label="ViQuAD (judge) vs step")
                with gr.Column():
                    vietnews_plot = gr.Plot(label="VietNews (judge) vs step")
            with gr.Row():
                with gr.Column():
                    vnhsge_plot = gr.Plot(label="VNHSGE-V vs step")
                with gr.Column():
                    vmlu_plot = gr.Plot(label="VMLU vs step")

            curve_summary = gr.Markdown(
                value="Summary will appear here.",
                label="Training curve summary",
            )

            # eval_all_curve_btn = gr.Button(
            #     "Eval ALL Benchmarks (ALL step_* checkpoints)"
            # )
            # all_bench_summary_tb = gr.Textbox(
            #     label="ALL Benchmarks over checkpoints summary",
            #     value="Click the button above to run.",
            #     lines=18,
            # )
        
        
    # Wiring
    refresh_btn.click(ui_refresh_models, inputs=None, outputs=model_dd)

    # khi đổi model family → update danh sách checkpoint + path
    model_dd.change(
        ui_update_checkpoint_list,
        inputs=model_dd,
        outputs=[ckpt_dd, starting_ckpt_tb],
    )

    # khi đổi checkpoint folder → update path
    ckpt_dd.change(
        ui_update_ckpt_path,
        inputs=[model_dd, ckpt_dd],
        outputs=starting_ckpt_tb,
    )

    # load model từ model + checkpoint folder
    load_btn.click(
        ui_load_model,
        inputs=[model_dd, ckpt_dd],
        outputs=[
            load_status,
            starting_ckpt_tb,
            run_mmlu_btn,
            run_zalo_btn,
            run_viquad_btn,
            run_vietnews_btn,
            run_vnhsge_btn,
            run_vmlu_btn,
        ],
    )

    run_mmlu_btn.click(ui_run_single_mmlu, inputs=None, outputs=mmlu_results_tb)
    run_zalo_btn.click(ui_run_single_zalo, inputs=None, outputs=zalo_results_tb)
    run_viquad_btn.click(ui_run_single_viquad, inputs=None, outputs=viquad_results_tb)
    run_vietnews_btn.click(ui_run_single_vietnews, inputs=None, outputs=vietnews_results_tb)
    run_vnhsge_btn.click(ui_run_single_vnhsge, inputs=None, outputs=vnhsge_results_tb)
    run_vmlu_btn.click(ui_run_single_vmlu, inputs=None, outputs=vmlu_results_tb)


    # Tab 2: khi đổi model family → list checkpoints step_xxx
    model_curve_dd.change(
        ui_list_ckpts_for_model_curve,
        inputs=model_curve_dd,
        outputs=ckpt_list_tb,
    )
    
    # Tab 2: analyze training curve (MMLU overall vs step)
    # analyze_btn.click(
    #     ui_analyze_training_curve,
    #     inputs=model_curve_dd,
    #     outputs=[curve_plot, curve_summary],
    # )
    
    analyze_btn.click(
    ui_analyze_training_curve,
    inputs=model_curve_dd,
    outputs=[
        mmlu_plot,
        zalo_plot,
        viq_plot,
        vietnews_plot,
        vnhsge_plot,
        vmlu_plot,
        curve_summary,
    ],
)

    
if __name__ == "__main__":
    demo.launch(
        debug=True,
        show_error=True,
        share=True,
    )
