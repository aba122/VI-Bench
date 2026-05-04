import os
import json
import time
import re
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer, AutoModel

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/InternVL2_5-8B"
DEVICE = "cuda"
MAX_NEW_TOKENS = 2048
NUM_SEGMENTS = 64  # number of frames to sample from video

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_A_PATH = os.path.join(BASE_DIR, "Dataset", "Videos", "Split_A.json")
UNIFIED_PROMPT_PATH = os.path.join(BASE_DIR, "System_Prompts", "unified_prompt.txt")

OUTPUT_PATH = os.path.join(BASE_DIR, "results", "Intern-VL25.json")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─── Image/Video Processing ────────────────────────────────────────────────
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_video(video_path, num_segments=16, max_num=1, input_size=448):
    """Load video and return pixel values and patch list for InternVL2.5."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1

    # Compute uniform frame indices
    seg_size = float(max_frame) / num_segments
    frame_indices = np.array([
        int(seg_size / 2 + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    frame_indices = np.clip(frame_indices, 0, max_frame)

    transform = build_transform(input_size=input_size)
    pixel_values_list = []
    num_patches_list = []

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img_tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in img_tiles])
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# ─── Utilities ──────────────────────────────────────────────────────────────
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def call_model(model, tokenizer, system_prompt, video_path):
    """Call InternVL2.5 with video frames."""
    pixel_values, num_patches_list = load_video(video_path, num_segments=NUM_SEGMENTS)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    # Build video frame prefix
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = f"{system_prompt}\n\n{video_prefix}Here is an AI-generated video. Please analyze the frames and complete the task."

    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False
    )

    return response.strip()


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
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)

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
                lambda vp: call_model(model, tokenizer, unified_sys, vp), easy_video)
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
                    lambda vp: call_model(model, tokenizer, unified_sys, vp), medium_video)
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
                    lambda vp: call_model(model, tokenizer, unified_sys, vp), hard_video)
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

# CUDA_VISIBLE_DEVICES=0,1,2,3 python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/inference/Intern-VL25.py

# conda activate internvl