"""
Generate videos from inference output prompts using Hunyuan and Wan2.1 models.
Reads result JSON files (e.g., Qwen3-VL.json), generates videos from Easy/Medium/Hard output prompts,
and saves a single merged JSON with video paths. Each item's "model" field determines which
video model is used.

Usage:
  # Both models (default)
  python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/inference/Video-Generate.py --input /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/results/Video-LLaMA3.json --gpus 2

  # Only Hunyuan
  python Video-Generate.py --input results/Qwen3-VL.json --video-model hunyuan --gpus 0,1,2,3

  # Only Wan
  python Video-Generate.py --input results/Qwen3-VL.json --video-model wan --gpus 0,1,2,3
"""

import json
import os
import random
import argparse
import multiprocessing as mp
import time
import queue
import subprocess
import numpy as np

SPLIT_A_PATH = "/p/fzv6enresearch/xwl/Prompt_Inversion_Bench/Dataset/Videos/Split_A.json"
# Dimensions matching both Hunyuan and Wan output
NOISE_NUM_FRAMES = 81
NOISE_FPS        = 16
NOISE_HEIGHT     = 480
NOISE_WIDTH      = 832


# ─── Noise Video ───────────────────────────────────────────────────────────
def generate_noise_video(output_path,
                         num_frames=NOISE_NUM_FRAMES,
                         fps=NOISE_FPS,
                         height=NOISE_HEIGHT,
                         width=NOISE_WIDTH):
    """Write a random-noise MP4 as a placeholder for missing inference outputs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        rng = np.random.default_rng()
        for _ in range(num_frames):
            frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
    except ImportError:
        # Fallback: ffmpeg noise via /dev/urandom pipe
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-pixel_format', 'rgb24',
            '-video_size', f'{width}x{height}',
            '-framerate', str(fps),
            '-i', '/dev/urandom',
            '-frames:v', str(num_frames),
            '-pix_fmt', 'yuv420p',
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    return os.path.exists(output_path)


def generate_noise_placeholders(data, output_data, input_basename, gt_by_id):
    """Generate noise placeholder videos for all missing/ERROR inference entries."""
    id_to_out_idx = {item["id"]: i for i, item in enumerate(output_data)}
    results_base  = "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/results"
    noise_count   = 0

    def _video_exists(path):
        return isinstance(path, str) and os.path.exists(path)

    def _is_missing(v):
        return not v or (isinstance(v, str) and v.upper().startswith('ERROR'))

    for item in data:
        item_id      = item["id"]
        model_lc     = item.get("model", "").lower()   # "wan" or "hunyuan"
        display_name = "Wan" if model_lc == "wan" else "Hunyuan"
        video_dir    = os.path.join(results_base, input_basename, display_name)
        out_idx      = id_to_out_idx[item_id]
        out          = output_data[out_idx]

        # ── Easy ──
        if _is_missing(item.get("Easy_Output")) and not _video_exists(out.get("Easy_Video")):
            path = os.path.join(video_dir, f"{item_id}_easy.mp4")
            if generate_noise_video(path):
                output_data[out_idx]["Easy_Video"] = path
                noise_count += 1
                print(f"  [noise] ID {item_id} Easy → {path}")

        # ── Medium ──
        if _is_missing(item.get("Medium_Output")) and not _video_exists(out.get("Medium_Video")):
            path = os.path.join(video_dir, f"{item_id}_medium.mp4")
            if generate_noise_video(path):
                output_data[out_idx]["Medium_Video"] = path
                noise_count += 1
                print(f"  [noise] ID {item_id} Medium → {path}")

        # ── Hard ──
        hard_prompts = item.get("Hard_Output")
        hard_missing = not isinstance(hard_prompts, list) or not hard_prompts
        if hard_missing:
            gt      = gt_by_id.get(item_id, {})
            n_shots = len(gt.get("hard", {}).get("hard_prompt", {})) or 1
            for shot_idx in range(n_shots):
                key  = f"Hard_Video_shot{shot_idx + 1}"
                if not _video_exists(out.get(key)):
                    path = os.path.join(video_dir, f"{item_id}_hard_shot{shot_idx + 1}.mp4")
                    if generate_noise_video(path):
                        output_data[out_idx][key] = path
                        noise_count += 1
                        print(f"  [noise] ID {item_id} Hard shot{shot_idx+1} → {path}")

    print(f"Noise placeholders generated: {noise_count}")
    return noise_count


def load_json(json_path: str) -> list:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: list, json_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── Model Configurations ──────────────────────────────────────────────────
MODEL_CONFIGS = {
    "hunyuan": {
        "model_path": "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/Hunyuan-T2V-420p/",
        "dit_ckpt": "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/Hunyuan-Distill-Models-T2V-420p/hy1.5_t2v_480p_lightx2v_4step.safetensors",
        "model_cls": "hunyuan_video_1.5",
        "transformer_model_name": "480p_t2v",
        "generator_kwargs": {
            "attn_mode": "flash_attn2",
            "infer_steps": 4,
            "num_frames": 81,
            "guidance_scale": 1,
            "sample_shift": 9.0,
            "aspect_ratio": "16:9",
            "fps": 16,
            "denoising_step_list": [1000, 750, 500, 250],
        },
        "negative_prompt": "",
    },
    "wan": {
        "model_path": "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/Wan2.1-T2V-14B",
        "dit_ckpt": "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/Wan2.1-Distill-Models/wan2.1_t2v_14b_lightx2v_4step.safetensors",
        "model_cls": "wan2.1_distill",
        "transformer_model_name": None,
        "generator_kwargs": {
            "attn_mode": "flash_attn2",
            "infer_steps": 4,
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "guidance_scale": 1,
            "sample_shift": 5.0,
            "denoising_step_list": [1000, 750, 500, 250],
        },
        "negative_prompt": "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    },
}


# ─── Task Building ─────────────────────────────────────────────────────────
def build_tasks(data, output_video_dir, video_model_name, difficulty='all'):
    """Build a flat list of generation tasks from inference results.

    Only includes items whose "model" field matches video_model_name.
    The difficulty parameter filters which levels to include: 'easy', 'medium', 'hard', or 'all'.
    """
    tasks = []
    for item in data:
        if item.get("model", "").lower() != video_model_name:
            continue
        item_id = item["id"]

        if difficulty in ('easy', 'all'):
            # Easy — Output is a list of shots; use first shot as prompt
            easy_raw = item.get("Easy_Output")
            easy_prompt = easy_raw[0] if isinstance(easy_raw, list) and easy_raw else easy_raw
            if easy_prompt and isinstance(easy_prompt, str) and not easy_prompt.startswith("ERROR"):
                tasks.append({
                    "id": item_id,
                    "difficulty": "Easy",
                    "prompt": easy_prompt,
                    "video_path": os.path.join(output_video_dir, f"{item_id}_easy.mp4"),
                    "video_key": "Easy_Video",
                })

        if difficulty in ('medium', 'all'):
            # Medium — Output is a list of shots; use first shot as prompt
            medium_raw = item.get("Medium_Output")
            medium_prompt = medium_raw[0] if isinstance(medium_raw, list) and medium_raw else medium_raw
            if medium_prompt and isinstance(medium_prompt, str) and not medium_prompt.startswith("ERROR"):
                tasks.append({
                    "id": item_id,
                    "difficulty": "Medium",
                    "prompt": medium_prompt,
                    "video_path": os.path.join(output_video_dir, f"{item_id}_medium.mp4"),
                    "video_key": "Medium_Video",
                })

        if difficulty in ('hard', 'all'):
            # Hard — each shot gets its own video
            hard_prompts = item.get("Hard_Output")
            if isinstance(hard_prompts, list):
                for shot_idx, shot_prompt in enumerate(hard_prompts):
                    if shot_prompt and not str(shot_prompt).startswith("ERROR"):
                        tasks.append({
                            "id": item_id,
                            "difficulty": "Hard",
                            "shot": shot_idx + 1,
                            "prompt": shot_prompt,
                            "video_path": os.path.join(output_video_dir, f"{item_id}_hard_shot{shot_idx+1}.mp4"),
                            "video_key": f"Hard_Video_shot{shot_idx+1}",
                        })

    return tasks


def get_completed_videos(output_data):
    """Extract set of completed video paths from output data."""
    completed = set()
    for item in output_data:
        for key, val in item.items():
            if key.endswith("_Video") or key.startswith("Hard_Video_shot"):
                if isinstance(val, str) and os.path.exists(val):
                    completed.add(val)
    return completed


# ─── GPU Worker ────────────────────────────────────────────────────────────
def gpu_worker(gpu_id, task_queue, result_queue, video_model_name):
    """Worker process: init model once, process tasks from queue."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    lightx2v_dir = "/p/fzv6enresearch/xwl/Prompt_Inversion_Bench/model/LightX2V"
    os.chdir(lightx2v_dir)
    import sys
    if lightx2v_dir not in sys.path:
        sys.path.insert(0, lightx2v_dir)

    config = MODEL_CONFIGS[video_model_name]

    try:
        from lightx2v import LightX2VPipeline

        print(f"[GPU {gpu_id}] Initializing {video_model_name} model...")

        pipe_kwargs = {
            "model_path": config["model_path"],
            "model_cls": config["model_cls"],
            "task": "t2v",
            "dit_original_ckpt": config["dit_ckpt"],
        }
        if config["transformer_model_name"]:
            pipe_kwargs["transformer_model_name"] = config["transformer_model_name"]

        pipe = LightX2VPipeline(**pipe_kwargs)
        pipe.enable_offload(
            cpu_offload=False,
            offload_granularity="block",
            text_encoder_offload=False,
            image_encoder_offload=False,
            vae_offload=False,
        )
        pipe.create_generator(**config["generator_kwargs"])
        print(f"[GPU {gpu_id}] Model initialized successfully")

        negative_prompt = config["negative_prompt"]

        while True:
            task = task_queue.get()
            if task is None:
                print(f"[GPU {gpu_id}] Received stop signal, exiting...")
                break

            seed = random.randint(0, 2**31 - 1)
            print(f"[GPU {gpu_id}] Generating {task['difficulty']} video for ID {task['id']}: {task['prompt'][:60]}...")

            try:
                pipe.generate(
                    seed=seed,
                    prompt=task['prompt'],
                    negative_prompt=negative_prompt,
                    save_result_path=task['video_path'],
                )
                if os.path.exists(task['video_path']):
                    print(f"[GPU {gpu_id}] Completed {task['difficulty']} video for ID {task['id']}")
                    result_queue.put({'success': True, 'task': task, 'seed': seed})
                else:
                    result_queue.put({'success': False, 'task': task, 'error': 'Video file not created'})
            except Exception as e:
                import traceback
                print(f"[GPU {gpu_id}] Error: {e}")
                traceback.print_exc()
                result_queue.put({'success': False, 'task': task, 'error': str(e)})

    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error during initialization: {e}")
        import traceback
        traceback.print_exc()
        while True:
            try:
                task = task_queue.get_nowait()
                if task is None:
                    break
                result_queue.put({'success': False, 'task': task, 'error': str(e)})
            except:
                break


# ─── Process One Video Model ──────────────────────────────────────────────
def process_video_model(video_model_name, data, input_basename, output_data, output_json_path, gpu_ids, checkpoint_interval, difficulty='all'):
    """Run generation for a single video model."""
    display_name = video_model_name.capitalize()
    output_video_dir = f"/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/results/{input_basename}/{display_name}"

    os.makedirs(output_video_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating videos with {display_name}")
    print(f"Video output: {output_video_dir}")
    print(f"JSON output:  {output_json_path}")
    print(f"{'='*60}")

    completed_videos = get_completed_videos(output_data)

    # Build and filter tasks (only items matching this video model)
    all_tasks = build_tasks(data, output_video_dir, video_model_name, difficulty=difficulty)
    pending_tasks = [t for t in all_tasks if t['video_path'] not in completed_videos]
    print(f"Total tasks: {len(all_tasks)}, Pending: {len(pending_tasks)}")

    if not pending_tasks:
        print("All videos already generated!")
        return

    id_to_idx = {item["id"]: i for i, item in enumerate(output_data)}

    # Multi-GPU
    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    workers = []
    for gpu_id in gpu_ids:
        p = ctx.Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue, video_model_name))
        p.start()
        workers.append(p)

    print("Waiting for workers to initialize...")
    time.sleep(5)

    for task in pending_tasks:
        task_queue.put(task)
    for _ in workers:
        task_queue.put(None)

    completed_count = 0
    failed_count = 0
    total_pending = len(pending_tasks)

    while completed_count + failed_count < total_pending:
        try:
            timeout = 1800 if completed_count == 0 else 900
            result = result_queue.get(timeout=timeout)
            task = result['task']
            out_idx = id_to_idx[task['id']]

            if result['success']:
                completed_count += 1
                output_data[out_idx][task['video_key']] = task['video_path']

                if completed_count % checkpoint_interval == 0:
                    save_json(output_data, output_json_path)
                    print(f"[{display_name}] Checkpoint: {completed_count}/{total_pending} done, {failed_count} failed")
            else:
                failed_count += 1
                print(f"[{display_name}] Failed: ID {task['id']} {task['difficulty']} - {result.get('error', 'Unknown')}")

        except queue.Empty:
            alive = [p for p in workers if p.is_alive()]
            if not alive:
                break
            print(f"Timeout, but {len(alive)} workers still running...")
        except Exception as e:
            print(f"Error: {e}")
            if not any(p.is_alive() for p in workers):
                break

    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    save_json(output_data, output_json_path)
    print(f"[{display_name}] Done! Completed: {completed_count}, Failed: {failed_count}")
    print(f"[{display_name}] Output: {output_json_path}")


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generate videos from inference output prompts')
    parser.add_argument('--input', type=str, required=True,
                        help='Input result JSON file (e.g., results/Qwen3-VL.json)')
    parser.add_argument('--video-model', type=str, choices=['hunyuan', 'wan', 'both'], default='both',
                        help='Which video generation model to use (default: both)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs (e.g., "0,1,2,3"). Default: all available')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='Save checkpoint every N completed videos')
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'hard', 'all'], default='all',
                        help='Which difficulty videos to generate (default: all)')
    args = parser.parse_args()

    # GPU IDs
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        import torch
        gpu_ids = list(range(torch.cuda.device_count()))
    print(f"Using GPUs: {gpu_ids}")

    # Load input data
    data = load_json(args.input)
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    results_dir = os.path.dirname(args.input)
    print(f"Loaded {len(data)} items from {args.input}")

    # Single merged output JSON under Video/ subdirectory
    video_results_dir = os.path.join(results_dir, "Video")
    os.makedirs(video_results_dir, exist_ok=True)
    output_json_path = os.path.join(video_results_dir, f"{input_basename}_Video.json")

    # Load or init output data
    if os.path.exists(output_json_path):
        output_data = load_json(output_json_path)
        print(f"Resumed from {output_json_path}: {len(output_data)} items")
        # Merge any new items from data that are missing in output_data
        existing_ids = {item["id"] for item in output_data}
        new_items = [item.copy() for item in data if item["id"] not in existing_ids]
        if new_items:
            output_data.extend(new_items)
            print(f"Added {len(new_items)} new items from inference JSON")
    else:
        output_data = [item.copy() for item in data]

    # ── Noise placeholders for missing/ERROR entries ──
    with open(SPLIT_A_PATH, 'r', encoding='utf-8') as f:
        split_a = json.load(f)
    gt_by_id = {item["id"]: item for item in split_a}

    print("\nGenerating noise placeholders for missing/ERROR entries ...")
    n_noise = generate_noise_placeholders(data, output_data, input_basename, gt_by_id)
    if n_noise:
        save_json(output_data, output_json_path)
        print(f"Saved {n_noise} noise placeholder paths to {output_json_path}")

    # Determine which models to run
    if args.video_model == 'both':
        video_models = ['hunyuan', 'wan']
    else:
        video_models = [args.video_model]

    for vm in video_models:
        process_video_model(vm, data, input_basename, output_data, output_json_path, gpu_ids, args.checkpoint_interval, difficulty=args.difficulty)

    print(f"\nAll done!")


if __name__ == "__main__":
    main()

# 使用示例:
# 两个模型都跑 (输出合并到一个 {name}_Video.json)
# python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/inference/Video-Generate.py --input /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/results/Qwen3-VL.json --gpus 0,1,2,3

# 只跑 Hunyuan
# python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/inference/Video-Generate.py --input /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/results/Qwen3-VL.json --video-model hunyuan --gpus 0,1,2,3

# 只跑 Wan
# python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/inference/Video-Generate.py --input /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/results/Qwen3-VL.json --video-model wan --gpus 0,1,2,3
