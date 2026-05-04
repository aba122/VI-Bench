"""
Evaluate CLIP-T (text-level) similarity between predicted prompts and ground-truth prompts,
plus Shots_Output accuracy.

Compares:
  - Easy:   Easy_Output       vs  easy.prompt
  - Medium: Medium_Output     vs  medium.rewrite_prompt
  - Hard:   Hard_Output[i]    vs  hard.hard_prompt.shot_{i+1}  (average across shots)
  - Shots accuracy: Shots_Output == hard.shots

Usage:
  python CLIP-I_shots.py --result-json results/Qwen25-VL.json
"""

import os
import json
import argparse
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer
from datetime import datetime


# ─── Configuration ──────────────────────────────────────────────────────────
CLIP_MODEL_PATH = "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/clip-vit-base-patch32"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_A_PATH = os.path.join(BASE_DIR, "Dataset", "Videos", "Split_A.json")
LOG_PATH = os.path.join(BASE_DIR, "Experiment.log")


# ─── Utilities ──────────────────────────────────────────────────────────────
def compute_text_similarity(model, tokenizer, text_a, text_b, device):
    """Compute CLIP cosine similarity between two text strings."""
    with torch.no_grad():
        inputs_a = tokenizer(text_a, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        inputs_b = tokenizer(text_b, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        feat_a = model.get_text_features(**inputs_a)
        feat_b = model.get_text_features(**inputs_b)
        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
        sim = (feat_a * feat_b).sum().item()
    return sim


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CLIP-T similarity + Shots accuracy evaluation")
    parser.add_argument("--result-json", type=str, required=True,
                        help="Path to the inference result JSON (e.g., results/Qwen25-VL.json)")
    args = parser.parse_args()

    # Derive model name from filename: Qwen25-VL.json -> Qwen25-VL
    eval_model_name = os.path.splitext(os.path.basename(args.result_json))[0]

    # Load data
    with open(args.result_json, "r", encoding="utf-8") as f:
        result_data = json.load(f)
    with open(SPLIT_A_PATH, "r", encoding="utf-8") as f:
        split_a = json.load(f)

    # Build id -> Split_A item lookup
    gt_by_id = {item["id"]: item for item in split_a}

    # Load CLIP model (text encoder only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model from {CLIP_MODEL_PATH} ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)
    print("CLIP model loaded.")

    easy_sims = []
    medium_sims = []
    hard_sims = []
    shots_correct = 0
    shots_total = 0

    total = len(result_data)
    for idx, item in enumerate(result_data):
        item_id = item["id"]
        gt = gt_by_id.get(item_id)
        if gt is None:
            print(f"[{idx+1}/{total}] ID {item_id}: not found in Split_A, skipping.")
            continue

        print(f"[{idx+1}/{total}] ID {item_id} ...", end="", flush=True)

        # ── Easy ──
        pred_easy = item.get("Easy_Output")
        gt_easy = gt["easy"]["prompt"]
        if pred_easy and isinstance(pred_easy, str) and not pred_easy.startswith("ERROR"):
            try:
                sim = compute_text_similarity(model, tokenizer, pred_easy, gt_easy, device)
                easy_sims.append(sim)
            except Exception as e:
                print(f" Easy err: {e}", end="")

        # ── Medium ──
        pred_medium = item.get("Medium_Output")
        gt_medium = gt["medium"]["rewrite_prompt"]
        if pred_medium and isinstance(pred_medium, str) and not pred_medium.startswith("ERROR"):
            try:
                sim = compute_text_similarity(model, tokenizer, pred_medium, gt_medium, device)
                medium_sims.append(sim)
            except Exception as e:
                print(f" Medium err: {e}", end="")

        # ── Hard (average across shots) ──
        pred_hard = item.get("Hard_Output")
        gt_hard_prompt = gt["hard"]["hard_prompt"]
        gt_shots = gt["hard"]["shots"]
        if isinstance(pred_hard, list):
            shot_sims = []
            for shot_idx, pred_shot in enumerate(pred_hard):
                gt_key = f"shot_{shot_idx + 1}"
                gt_shot = gt_hard_prompt.get(gt_key)
                if gt_shot and pred_shot and isinstance(pred_shot, str) and not pred_shot.startswith("ERROR"):
                    try:
                        sim = compute_text_similarity(model, tokenizer, pred_shot, gt_shot, device)
                        shot_sims.append(sim)
                    except Exception as e:
                        print(f" Hard shot{shot_idx+1} err: {e}", end="")
            if shot_sims:
                hard_sims.append(np.mean(shot_sims))

        # ── Shots accuracy ──
        pred_shots = item.get("Shots_Output")
        if isinstance(pred_shots, int) and isinstance(gt_shots, int):
            shots_total += 1
            if pred_shots == gt_shots:
                shots_correct += 1

        print(" done")

    # ── Compute metrics ──
    easy_avg = np.mean(easy_sims) if easy_sims else 0.0
    medium_avg = np.mean(medium_sims) if medium_sims else 0.0
    hard_avg = np.mean(hard_sims) if hard_sims else 0.0
    overall_avg = (easy_avg + medium_avg + hard_avg) / 3.0
    shots_acc = (shots_correct / shots_total * 100) if shots_total > 0 else 0.0

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"CLIP-T & Shots Evaluation Results for: {eval_model_name}")
    print(f"{'='*60}")
    print(f"  Easy   CLIP-T:  {easy_avg:.4f}  (n={len(easy_sims)})")
    print(f"  Medium CLIP-T:  {medium_avg:.4f}  (n={len(medium_sims)})")
    print(f"  Hard   CLIP-T:  {hard_avg:.4f}  (n={len(hard_sims)})")
    print(f"  Overall CLIP-T: {overall_avg:.4f}")
    print(f"  Shots Accuracy: {shots_acc:.2f}%  ({shots_correct}/{shots_total})")
    print(f"{'='*60}")

    # ── Append to Experiment.log ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [
        f"\n[{timestamp}] CLIP-T & Shots Evaluation — Model: {eval_model_name}",
        f"  Easy   CLIP-T:  {easy_avg:.4f}  (n={len(easy_sims)})",
        f"  Medium CLIP-T:  {medium_avg:.4f}  (n={len(medium_sims)})",
        f"  Hard   CLIP-T:  {hard_avg:.4f}  (n={len(hard_sims)})",
        f"  Overall CLIP-T: {overall_avg:.4f}",
        f"  Shots Accuracy: {shots_acc:.2f}%  ({shots_correct}/{shots_total})",
        "",
    ]
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Results appended to {LOG_PATH}")


if __name__ == "__main__":
    main()

# conda activate Hunyuan
# CUDA_VISIBLE_DEVICES=0 python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/evaluate/CLIP-T_shots.py --result-json /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/results/Qwen25-VL.json
