#!/usr/bin/env python3
"""
semantic_unit_recovery.py
=========================
Analyse which semantic units from GT prompts are recovered by VLMs.

Phase 1 (--phase extract):
  Load Qwen3-VL-30B on CUDA, extract semantic units from all GT prompts (text-only).
  Cache → results/semantic_units/gt_units.json

Phase 2 (--phase judge):
  Load Qwen3-VL-30B on CUDA, judge recovery of semantic units in inversion prompts.
  Cache → results/semantic_units/{model}_judgments.json

Usage:
  CUDA_VISIBLE_DEVICES=2,3 python evaluate/semantic_unit_recovery.py --phase extract
  CUDA_VISIBLE_DEVICES=2,3 python evaluate/semantic_unit_recovery.py --phase judge [--model Qwen3-VL]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
SPLIT_A    = BASE_DIR / "Dataset" / "Videos" / "Split_A.json"
RESULTS    = BASE_DIR / "results"
OUT_DIR    = RESULTS / "semantic_units"
OUT_DIR.mkdir(exist_ok=True)

GT_UNITS_CACHE = OUT_DIR / "gt_units.json"

MODEL_30B  = "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/Qwen3-VL-30B-A3B-Instruct"

# (GPT-4o not used; judgment is done by Qwen3-VL-30B)

# ── VLM models to judge ───────────────────────────────────────────────────────
EXCLUDE = {"sft_example_shot_detection", "M3-Agent", "Qwen3.5"}
DIFFICULTIES = ["Easy", "Medium", "Hard"]

CATEGORIES = ["Subject", "Action", "Scene", "Style", "Camera"]

# ─────────────────────────────────────────────────────────────────────────────
# Shared model-call helper (text-only)
# ─────────────────────────────────────────────────────────────────────────────

def call_model_text(model, processor, system_prompt, user_prompt, max_new_tokens=1024):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[chat_text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = gen_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Extract semantic units with Qwen3-VL-30B (text-only)
# ─────────────────────────────────────────────────────────────────────────────

EXTRACT_SYSTEM = """You are a video-prompt semantic analyst. Given a text-to-video prompt, \
extract all key semantic units (1–6 word phrases) that describe visually distinct content. \
Classify each unit into exactly one of:
  Subject  – main entities: people, animals, objects, characters
  Action   – what subjects do: motions, gestures, interactions, events
  Scene    – setting, environment, background, lighting, atmosphere
  Style    – visual aesthetics, rendering, cinematic quality, color palette
  Camera   – shot type, camera movement, angle, framing, lens

Rules:
- Extract 8–20 units per prompt; prioritise specificity over generality
- Phrases must be 1–6 words, all lowercase, no punctuation
- Avoid duplicates; each unit must convey distinct information
- Return ONLY a JSON array: [{"phrase":"...","category":"..."},...]"""

EXTRACT_USER = "Extract semantic units from this video prompt:\n\n{prompt}"


def extract_units_batch(model, processor, prompts_batch):
    """
    Run Qwen3-VL-30B on a batch of (key, text) pairs.
    Returns dict: key → list[{phrase,category}]
    """
    results = {}
    for key, text in prompts_batch:
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user",   "content": EXTRACT_USER.format(prompt=text)},
        ]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = processor(text=[chat_text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        trimmed = gen_ids[0][inputs.input_ids.shape[1]:]
        raw = processor.decode(trimmed, skip_special_tokens=True).strip()

        # Parse JSON
        try:
            m = re.search(r"\[[\s\S]*\]", raw)
            units = json.loads(m.group()) if m else []
            # Normalise category
            valid = []
            for u in units:
                cat = u.get("category", "").strip()
                if cat not in CATEGORIES:
                    # Fuzzy match
                    cat_lower = cat.lower()
                    for c in CATEGORIES:
                        if c.lower() in cat_lower or cat_lower in c.lower():
                            cat = c
                            break
                    else:
                        cat = "Scene"
                valid.append({"phrase": u.get("phrase","").lower().strip(), "category": cat})
            results[key] = valid
        except Exception as e:
            print(f"  [WARN] Parse failed for {key}: {e} | raw={raw[:100]}")
            results[key] = []
    return results


def phase_extract():
    print("=== Phase 1: Extract semantic units (Qwen3-VL-30B) ===")

    with open(SPLIT_A) as f:
        data = json.load(f)

    # Load existing cache
    cache = {}
    if GT_UNITS_CACHE.exists():
        with open(GT_UNITS_CACHE) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached entries")

    # Build list of (key, prompt_text) to process
    to_process = []
    for item in data:
        iid = item["id"]
        # Easy
        key_e = f"easy_{iid}"
        if key_e not in cache:
            p = item["easy"].get("prompt", "")
            if p: to_process.append((key_e, p))
        # Medium
        key_m = f"medium_{iid}"
        if key_m not in cache:
            p = item["medium"].get("rewrite_prompt", "")
            if p: to_process.append((key_m, p))
        # Hard: each shot separately, then aggregate key
        hp = item["hard"].get("hard_prompt", {})
        shots = [v for v in hp.values() if isinstance(v, str) and v]
        for si, shot_text in enumerate(shots, 1):
            key_s = f"hard_{iid}_shot{si}"
            if key_s not in cache:
                to_process.append((key_s, shot_text))

    if not to_process:
        print("All entries already cached.")
        return

    print(f"Need to extract {len(to_process)} prompts")
    print("Loading Qwen3-VL-30B-A3B-Instruct ...")

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_30B,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_30B)

    print("Model loaded. Extracting ...")
    batch_size = 1  # process one at a time for safety
    for i in tqdm(range(0, len(to_process), batch_size)):
        batch = to_process[i:i + batch_size]
        try:
            res = extract_units_batch(model, processor, batch)
            cache.update(res)
        except Exception as e:
            print(f"  [ERROR] Batch {i}: {e}")
            for key, _ in batch:
                cache[key] = []

        # Save checkpoint every 10
        if (i // batch_size) % 10 == 0:
            with open(GT_UNITS_CACHE, "w") as f:
                json.dump(cache, f, indent=2)

    with open(GT_UNITS_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Saved {len(cache)} entries → {GT_UNITS_CACHE}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Judge recovery with Qwen3-VL-30B (text-only)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are evaluating whether semantic units from a GT video prompt \
are recovered in a VLM-generated inversion prompt.

Score each unit:
  2 = Fully recovered – the same or equivalent concept is clearly present
  1 = Partially recovered – similar idea present but weaker/vaguer/incomplete
  0 = Missing – not mentioned, contradicted, or substituted with wrong meaning

Return ONLY a JSON object mapping each phrase to its integer score: {"phrase": score, ...}
Include ALL provided phrases."""

JUDGE_USER = """GT semantic units to check:
{units_str}

VLM inversion prompt:
{inv_prompt}"""


def judge_one_model(model, processor, units, inv_prompt):
    """Judge recovery using Qwen3-VL-30B. Returns {phrase: score}."""
    if not units or not inv_prompt.strip():
        return {u["phrase"]: 0 for u in units}

    units_str = "\n".join(f'- "{u["phrase"]}" [{u["category"]}]' for u in units)
    raw = call_model_text(
        model, processor, JUDGE_SYSTEM,
        JUDGE_USER.format(units_str=units_str, inv_prompt=inv_prompt[:1500]),
        max_new_tokens=512,
    )
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        scores = json.loads(m.group()) if m else {}
        result = {}
        for u in units:
            phrase = u["phrase"]
            score = scores.get(phrase, None)
            if score is None:
                for k, v in scores.items():
                    if k.lower() == phrase.lower():
                        score = v
                        break
            result[phrase] = int(score) if score is not None else 0
        return result
    except Exception as e:
        print(f"  [WARN] Judge parse error: {e} | raw={raw[:100]}")
        return {u["phrase"]: 0 for u in units}


def get_gt_units_for_difficulty(cache, iid, difficulty):
    if difficulty == "Easy":
        return cache.get(f"easy_{iid}", [])
    elif difficulty == "Medium":
        return cache.get(f"medium_{iid}", [])
    else:
        units_all, seen = [], set()
        for si in range(1, 6):
            for u in cache.get(f"hard_{iid}_shot{si}", []):
                if u["phrase"] not in seen:
                    units_all.append(u)
                    seen.add(u["phrase"])
        return units_all


def get_inv_prompt_for_difficulty(item_result, difficulty):
    if difficulty == "Easy":
        out = item_result.get("Easy_Output", [])
        return out[0] if out else ""
    elif difficulty == "Medium":
        out = item_result.get("Medium_Output", [])
        return out[0] if out else ""
    else:
        out = item_result.get("Hard_Output", [])
        return " ".join(out) if out else ""


def phase_judge(target_model=None):
    print("=== Phase 2: Judge recovery (Qwen3-VL-30B) ===")

    if not GT_UNITS_CACHE.exists():
        print("ERROR: Run --phase extract first.")
        sys.exit(1)

    with open(GT_UNITS_CACHE) as f:
        gt_cache = json.load(f)
    print(f"Loaded {len(gt_cache)} GT unit entries")

    # Find all model result files
    model_files = []
    for fp in sorted(RESULTS.glob("*.json")):
        name = fp.stem
        if "rank" in name or name in EXCLUDE:
            continue
        if target_model and name != target_model:
            continue
        model_files.append((name, fp))

    print(f"Models to process: {[m for m, _ in model_files]}")

    print("Loading Qwen3-VL-30B-A3B-Instruct ...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_30B, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_30B)
    print("Model loaded.")

    for model_name, fp in model_files:
        out_path = OUT_DIR / f"{model_name}_judgments.json"

        judgments = {}
        if out_path.exists():
            with open(out_path) as f:
                judgments = json.load(f)

        with open(fp) as f:
            model_results = {item["id"]: item for item in json.load(f)}

        # Build task list
        tasks = []
        for iid, item_result in model_results.items():
            for diff in DIFFICULTIES:
                key = f"{diff}_{iid}"
                if key in judgments:
                    continue
                units = get_gt_units_for_difficulty(gt_cache, iid, diff)
                if not units:
                    continue
                inv = get_inv_prompt_for_difficulty(item_result, diff)
                if not inv:
                    continue
                tasks.append((key, units, inv))

        if not tasks:
            print(f"  {model_name}: all {len(judgments)} cached, skipping")
            continue

        print(f"  {model_name}: {len(judgments)} cached, {len(tasks)} to judge")

        for i, (key, units, inv) in enumerate(tqdm(tasks, desc=model_name)):
            try:
                scores = judge_one_model(model, processor, units, inv)
                judgments[key] = {"units": units, "scores": scores}
            except Exception as e:
                print(f"  [ERROR] {key}: {e}")
                judgments[key] = {"units": units, "scores": {u["phrase"]: 0 for u in units}}

            if i % 100 == 0:
                with open(out_path, "w") as f:
                    json.dump(judgments, f)

        with open(out_path, "w") as f:
            json.dump(judgments, f, indent=2)
        print(f"  {model_name}: saved {len(judgments)} → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["extract", "judge"], required=True)
    parser.add_argument("--model", default=None, help="Judge only this model")
    args = parser.parse_args()

    if args.phase == "extract":
        phase_extract()
    elif args.phase == "judge":
        phase_judge(target_model=args.model)


if __name__ == "__main__":
    main()
