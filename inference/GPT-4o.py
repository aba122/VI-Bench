import os
import json
import base64
import http.client
import time
import cv2
import re

# ─── Configuration ──────────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "YOUR_API_KEY")
API_HOST = "api.302.ai"
MODEL = "gpt-4o-2024-11-20"
NUM_FRAMES = 16
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_A_PATH = os.path.join(BASE_DIR, "Dataset", "Videos", "Split_A.json")
UNIFIED_PROMPT_PATH = os.path.join(BASE_DIR, "System_Prompts", "unified_prompt.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "GPT-4o.json")


# ─── Utilities ──────────────────────────────────────────────────────────────
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def sample_frames_base64(video_path, num_frames=16):
    """Uniformly sample `num_frames` frames from a video and return as base64 JPEG strings."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    # Compute uniform frame indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        frames_b64.append(base64.b64encode(buffer).decode("utf-8"))

    cap.release()
    return frames_b64


def call_model(system_prompt, frames_b64):
    """Call GPT-4o API with system prompt and base64-encoded frames."""
    user_content = [{"type": "text", "text": "Here is an AI-generated video. Please analyze it and complete the task."}]
    for b64 in frames_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })

    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 2048,
        "temperature": 0.2
    })

    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = http.client.HTTPSConnection(API_HOST, timeout=120)
            conn.request("POST", "/v1/chat/completions", payload, headers)
            response = conn.getresponse()
            data = json.loads(response.read().decode("utf-8"))
            conn.close()

            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            elif "error" in data:
                print(f"  [Attempt {attempt}/{MAX_RETRIES}] API error: {data['error']}")
            else:
                print(f"  [Attempt {attempt}/{MAX_RETRIES}] Unexpected response: {json.dumps(data)[:200]}")
        except Exception as e:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Exception: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("Max retries exceeded for API call")


def parse_output(raw_output):
    """Parse the JSON output to extract shots count and per-shot prompts."""
    json_match = re.search(r'\{[\s\S]*\}', raw_output)
    if not json_match:
        raise ValueError(f"Cannot parse JSON from output: {raw_output[:200]}")

    parsed = json.loads(json_match.group())
    shots = parsed.get("shots", 0)
    prompts = []
    for i in range(1, shots + 1):
        key = f"shot_{i}"
        if key in parsed:
            prompts.append(parsed[key])
    return shots, prompts


def infer_with_retry(frames_b64, system_prompt, max_retries=5):
    """Call model and parse JSON output, retry up to max_retries times on failure."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_model(system_prompt, frames_b64)
            shots, prompts = parse_output(raw)
            if attempt > 1:
                print(f"    Retry {attempt} succeeded.")
            return shots, prompts
        except Exception as e:
            last_err = e
            print(f"    Attempt {attempt}/{max_retries} failed: {e}")
    raise RuntimeError(f"Failed after {max_retries} attempts. Last: {last_err}")


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    with open(SPLIT_A_PATH, "r", encoding="utf-8") as f:
        split_a = json.load(f)

    unified_sys = load_system_prompt(UNIFIED_PROMPT_PATH)

    # Load existing results for resume support
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        print(f"Loaded {len(results)} existing results, resuming...")
    else:
        results = []
        completed_ids = set()

    total = len(split_a)
    for idx, sample in enumerate(split_a):
        sample_id = sample["id"]

        if sample_id in completed_ids:
            print(f"[{idx+1}/{total}] ID {sample_id} already done, skipping.")
            continue

        print(f"[{idx+1}/{total}] Processing ID {sample_id} (model: {sample['model']})...")
        result = {"id": sample_id, "model": sample["model"]}

        # ── Easy ──
        try:
            easy_video = sample["easy"]["video_path"]
            print(f"  Easy: {easy_video}")
            easy_frames = sample_frames_base64(easy_video, NUM_FRAMES)
            easy_shots, easy_prompts = infer_with_retry(easy_frames, unified_sys)
            result["Easy_Shots"] = easy_shots
            result["Easy_Output"] = easy_prompts
            print(f"  Easy done. Shots: {easy_shots}, prompts: {len(easy_prompts)}")
        except Exception as e:
            print(f"  Easy FAILED: {e}")
            result["Easy_Output"] = f"ERROR: {e}"

        # ── Medium ──
        try:
            medium_video = sample["medium"]["video_path"]
            print(f"  Medium: {medium_video}")
            medium_frames = sample_frames_base64(medium_video, NUM_FRAMES)
            medium_shots, medium_prompts = infer_with_retry(medium_frames, unified_sys)
            result["Medium_Shots"] = medium_shots
            result["Medium_Output"] = medium_prompts
            print(f"  Medium done. Shots: {medium_shots}, prompts: {len(medium_prompts)}")
        except Exception as e:
            print(f"  Medium FAILED: {e}")
            result["Medium_Output"] = f"ERROR: {e}"

        # ── Hard ──
        try:
            hard_video = sample["hard"]["video_path"]
            print(f"  Hard: {hard_video}")
            hard_frames = sample_frames_base64(hard_video, NUM_FRAMES)
            shots, hard_prompts = infer_with_retry(hard_frames, unified_sys)
            result["Shots_Output"] = shots
            result["Hard_Output"] = hard_prompts
            print(f"  Hard done. Shots: {shots}, prompts: {len(hard_prompts)}")
        except Exception as e:
            print(f"  Hard FAILED: {e}")
            result["Shots_Output"] = f"ERROR: {e}"
            result["Hard_Output"] = f"ERROR: {e}"

        results.append(result)
        completed_ids.add(sample_id)

        # Save after each sample for crash resilience
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved. Total completed: {len(results)}/{total}")

    print(f"\nDone! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
