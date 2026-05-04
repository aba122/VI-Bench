import os
import sys
import json
import time
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Ensure the conda env's bin (ffprobe, ffmpeg) is in PATH
_conda_bin = os.path.dirname(os.path.abspath(sys.executable))
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/VideoLLaMA3-7B"
DEVICE = "cuda"
MAX_NEW_TOKENS = 2048
NUM_FRAMES = 180

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_A_PATH = os.path.join(BASE_DIR, "Dataset", "Videos", "Split_A.json")
UNIFIED_PROMPT_PATH = os.path.join(BASE_DIR, "System_Prompts", "unified_prompt.txt")

OUTPUT_PATH = os.path.join(BASE_DIR, "results", "Video-LLaMA3.json")


# ─── Utilities ──────────────────────────────────────────────────────────────
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def call_model(model, processor, system_prompt, video_path):
    """Call VideoLLaMA3 with video input."""
    conversation = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": NUM_FRAMES}},
                {"type": "text", "text": "Here is an AI-generated video. Please analyze it and complete the task."},
            ],
        },
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # Decode full output and extract assistant response
    full_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    # Extract content after the last assistant marker
    marker = "<|im_start|>assistant\n"
    if marker in full_text:
        output_text = full_text.split(marker)[-1]
        output_text = output_text.replace("<|im_end|>", "").strip()
    else:
        output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output_text


def parse_hard_output(raw_output):
    """Parse the hard mode JSON output to extract shots count and per-shot prompts."""
    json_match = re.search(r'\{[\s\S]*\}', raw_output)
    if not json_match:
        raise ValueError(f"Cannot parse JSON from hard output: {raw_output[:200]}")

    parsed = json.loads(json_match.group())
    shots = parsed.get("shots", 0)
    hard_prompts = []
    for i in range(1, shots + 1):
        key = f"shot_{i}"
        if key in parsed:
            hard_prompts.append(parsed[key])
    return shots, hard_prompts


def infer_with_retry(call_fn, video_path, max_retries=5):
    """Call model and parse JSON output, retry up to max_retries times on failure."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_fn(video_path)
            shots, prompts = parse_hard_output(raw)
            if attempt > 1:
                print(f"    Retry {attempt} succeeded.")
            return shots, prompts
        except Exception as e:
            last_err = e
            print(f"    Attempt {attempt}/{max_retries} failed: {e}")
    raise RuntimeError(f"Failed after {max_retries} attempts. Last: {last_err}")


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--easy-only', action='store_true',
                        help='Re-run Easy only, preserve existing Medium/Hard results')
    args = parser.parse_args()

    print(f"Loading model: {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model loaded.")

    # Load data and prompts
    with open(SPLIT_A_PATH, "r", encoding="utf-8") as f:
        split_a = json.load(f)

    unified_sys = load_system_prompt(UNIFIED_PROMPT_PATH)

    # Load existing results for resume support
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        result_by_id = {r["id"]: r for r in results}
        if args.easy_only:
            completed_ids = {
                r["id"] for r in results
                if r.get("Easy_Output") and isinstance(r.get("Easy_Output"), list) and len(r.get("Easy_Output", [])) > 0
            }
        else:
            completed_ids = {r["id"] for r in results}
        print(f"Loaded {len(results)} existing results, resuming...")
    else:
        results = []
        result_by_id = {}
        completed_ids = set()

    total = len(split_a)
    for idx, sample in enumerate(split_a):
        sample_id = sample["id"]

        if sample_id in completed_ids:
            print(f"[{idx+1}/{total}] ID {sample_id} already done, skipping.")
            continue

        print(f"[{idx+1}/{total}] Processing ID {sample_id} (model: {sample['model']})...")
        if args.easy_only:
            result = result_by_id.get(sample_id, {"id": sample_id, "model": sample["model"]}).copy()
        else:
            result = {"id": sample_id, "model": sample["model"]}

        # ── Easy ──
        try:
            easy_video = sample["easy"]["video_path"]
            print(f"  Easy: {easy_video}")
            t0 = time.time()
            easy_shots, easy_prompts = infer_with_retry(
                lambda vp: call_model(model, processor, unified_sys, vp), easy_video)
            result["Easy_Shots"] = easy_shots
            result["Easy_Output"] = easy_prompts
            print(f"  Easy done ({time.time()-t0:.1f}s).")
        except Exception as e:
            print(f"  Easy FAILED: {e}")
            result["Easy_Output"] = f"ERROR: {e}"

        if not args.easy_only:
            # ── Medium ──
            try:
                medium_video = sample["medium"]["video_path"]
                print(f"  Medium: {medium_video}")
                t0 = time.time()
                medium_shots, medium_prompts = infer_with_retry(
                    lambda vp: call_model(model, processor, unified_sys, vp), medium_video)
                result["Medium_Shots"] = medium_shots
                result["Medium_Output"] = medium_prompts
                print(f"  Medium done ({time.time()-t0:.1f}s).")
            except Exception as e:
                print(f"  Medium FAILED: {e}")
                result["Medium_Output"] = f"ERROR: {e}"

        if not args.easy_only:
            # ── Hard ──
            try:
                hard_video = sample["hard"]["video_path"]
                print(f"  Hard: {hard_video}")
                t0 = time.time()
                shots, hard_prompts = infer_with_retry(
                    lambda vp: call_model(model, processor, unified_sys, vp), hard_video)
                result["Shots_Output"] = shots
                result["Hard_Output"] = hard_prompts
                print(f"  Hard done ({time.time()-t0:.1f}s). Shots: {shots}, prompts: {len(hard_prompts)}")
            except Exception as e:
                print(f"  Hard FAILED: {e}")
                result["Shots_Output"] = f"ERROR: {e}"
                result["Hard_Output"] = f"ERROR: {e}"

        if args.easy_only:
            existing_idx = next((i for i, r in enumerate(results) if r["id"] == sample_id), None)
            if existing_idx is not None:
                results[existing_idx] = result
            else:
                results.append(result)
                completed_ids.add(sample_id)
        else:
            results.append(result)
            completed_ids.add(sample_id)

        # Save after each sample for crash resilience
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved. Total completed: {len(results)}/{total}")

    print(f"\nDone! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0,1,2,3 python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/inference/Video-LLaMA3.py

# conda activate internvl