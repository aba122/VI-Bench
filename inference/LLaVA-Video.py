import os
import json
import time
import re
import argparse
import copy
import numpy as np
import torch
from decord import VideoReader, cpu

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/LLaVA-Video-7B"
DEVICE = "cuda"
MAX_NEW_TOKENS = 2048
MAX_FRAMES = 32
CONV_TEMPLATE = "qwen_1_5"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_A_PATH = os.path.join(BASE_DIR, "Dataset", "Videos", "Split_A.json")
UNIFIED_PROMPT_PATH = os.path.join(BASE_DIR, "System_Prompts", "unified_prompt.txt")

OUTPUT_PATH = os.path.join(BASE_DIR, "results", "LLaVA-Video.json")


# ─── Video Processing ──────────────────────────────────────────────────────
def load_video(video_path, max_frames_num=16):
    """Load video and uniformly sample frames."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()

    # Uniform sampling
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])

    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_time_str, video_time


# ─── Utilities ──────────────────────────────────────────────────────────────
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def call_model(model, tokenizer, image_processor, system_prompt, video_path):
    """Call LLaVA-Video with video frames."""
    frames, frame_time_str, video_time = load_video(video_path, max_frames_num=MAX_FRAMES)

    # Process frames
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video_tensor = [video_tensor]

    # Build prompt
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. "
        f"These frames are located at {frame_time_str}."
    )
    question = (
        f"{DEFAULT_IMAGE_TOKEN}\n{time_instruction}\n"
        f"{system_prompt}\n"
        f"Here is an AI-generated video. Please analyze it and complete the task."
    )

    conv = copy.deepcopy(conv_templates[CONV_TEMPLATE])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=video_tensor,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of this process for data-parallel inference')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Total number of parallel processes')
    args = parser.parse_args()

    # Rank-specific output path for data-parallel inference
    if args.world_size > 1:
        out_path = OUTPUT_PATH.replace('.json', f'_rank{args.rank}.json')
        print(f"[Rank {args.rank}/{args.world_size}] Output: {out_path}")
    else:
        out_path = OUTPUT_PATH

    print(f"Loading model: {MODEL_NAME} ...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        MODEL_NAME,
        None,
        "llava_qwen",
        torch_dtype="bfloat16",
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    # Load data and prompts
    with open(SPLIT_A_PATH, "r", encoding="utf-8") as f:
        split_a = json.load(f)

    unified_sys = load_system_prompt(UNIFIED_PROMPT_PATH)

    # Load existing results for resume support
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
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

    # Data-parallel: each rank handles a subset of samples
    if args.world_size > 1:
        split_a = [s for i, s in enumerate(split_a) if i % args.world_size == args.rank]
        print(f"[Rank {args.rank}] Handling {len(split_a)} samples")

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
                lambda vp: call_model(model, tokenizer, image_processor, unified_sys, vp), easy_video)
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
                    lambda vp: call_model(model, tokenizer, image_processor, unified_sys, vp), medium_video)
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
                    lambda vp: call_model(model, tokenizer, image_processor, unified_sys, vp), hard_video)
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
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved. Total completed: {len(results)}/{total}")

    print(f"\nDone! Results saved to {out_path}")


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/inference/LLaVA-Video.py

# conda activate internvl