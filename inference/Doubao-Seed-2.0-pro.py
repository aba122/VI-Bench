import os
import sys
import json
import base64
import time
import cv2
import re
import fcntl
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from volcenginesdkarkruntime import Ark

# ─── Configuration ──────────────────────────────────────────────────────────
ARK_API_KEY = os.getenv("ARK_API_KEY", "YOUR_API_KEY")
MODEL = "doubao-seed-2-0-pro-260215"
NUM_FRAMES = 8
MAX_RETRIES = 5
RETRY_DELAY = 5       # seconds between API retries
NUM_WORKERS = 5       # concurrent samples

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_A_PATH = os.path.join(BASE_DIR, "Dataset", "Videos", "Split_A.json")
UNIFIED_PROMPT_PATH = os.path.join(BASE_DIR, "System_Prompts", "unified_prompt.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "Doubao-Seed-2.0-pro.json")
BACKUP_PATH = "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/results/Doubao-Seed-2.0-pro.json"

client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY,
)

# Global stop signal: set when a fatal API error occurs
stop_event = threading.Event()


class FatalAPIError(Exception):
    """Unrecoverable API error (account overdue, hard rate limit). Stop immediately."""
    pass


# ─── Utilities ──────────────────────────────────────────────────────────────
def load_system_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def sample_frames_base64(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
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
    if stop_event.is_set():
        raise FatalAPIError("Stopped due to fatal error in another thread.")

    user_content = [
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
        for b64 in frames_b64
    ]
    user_content.append({
        "type": "input_text",
        "text": "Here is an AI-generated video. Please analyze it and complete the task."
    })

    for attempt in range(1, MAX_RETRIES + 1):
        if stop_event.is_set():
            raise FatalAPIError("Stopped due to fatal error in another thread.")
        try:
            response = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": user_content},
                ]
            )
            for item in response.output:
                if item.type == "message":
                    for part in item.content:
                        if part.type == "output_text" and part.text:
                            return part.text.strip()
            raise ValueError(f"No output_text in response: {response}")
        except FatalAPIError:
            raise
        except Exception as e:
            err_str = str(e)
            if "AccountOverdueError" in err_str or "SetLimitExceeded" in err_str:
                stop_event.set()
                raise FatalAPIError(f"Fatal API error: {e}")
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Exception: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError("Max retries exceeded for Ark API call")


def parse_output(raw_output):
    json_match = re.search(r'\{[\s\S]*\}', raw_output)
    if not json_match:
        raise ValueError(f"Cannot parse JSON from output: {raw_output[:200]}")
    parsed = json.loads(json_match.group())
    shots = parsed.get("shots", 0)
    prompts = [parsed[f"shot_{i}"] for i in range(1, shots + 1) if f"shot_{i}" in parsed]
    return shots, prompts


def infer_with_retry(frames_b64, system_prompt, max_retries=5):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_model(system_prompt, frames_b64)
            shots, prompts = parse_output(raw)
            if attempt > 1:
                print(f"    Retry {attempt} succeeded.")
            return shots, prompts
        except FatalAPIError:
            raise
        except Exception as e:
            last_err = e
            print(f"    Attempt {attempt}/{max_retries} failed: {e}")
    raise RuntimeError(f"Failed after {max_retries} attempts. Last: {last_err}")


def is_clean(entry):
    return isinstance(entry.get("Easy_Output"), list)


def save_results(results, path, lock):
    """Merge with on-disk file (always keep clean entries), then atomically write.
    Safe against concurrent NFS writes: clean entries are never overwritten by errors.
    """
    with lock:
        existing_by_id = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for r in json.load(f):
                        existing_by_id[r["id"]] = r
            except Exception:
                pass

        merged_by_id = dict(existing_by_id)
        for r in results:
            rid = r["id"]
            if is_clean(r) or not is_clean(merged_by_id.get(rid, {})):
                merged_by_id[rid] = r

        merged = sorted(merged_by_id.values(), key=lambda r: r["id"])
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)

        # Also save to local disk backup (not NFS)
        if BACKUP_PATH:
            try:
                os.makedirs(os.path.dirname(BACKUP_PATH), exist_ok=True)
                backup_tmp = BACKUP_PATH + ".tmp"
                with open(backup_tmp, "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False)
                os.replace(backup_tmp, BACKUP_PATH)
            except Exception as e:
                print(f"  [WARN] Backup save failed: {e}")

        return merged


# ─── Per-sample worker ──────────────────────────────────────────────────────
def process_sample(sample, unified_sys):
    sid = sample["id"]
    result = {"id": sid, "model": sample["model"]}
    try:
        # Easy
        easy_frames = sample_frames_base64(sample["easy"]["video_path"], NUM_FRAMES)
        easy_shots, easy_prompts = infer_with_retry(easy_frames, unified_sys)
        result["Easy_Shots"] = easy_shots
        result["Easy_Output"] = easy_prompts

        # Medium
        medium_frames = sample_frames_base64(sample["medium"]["video_path"], NUM_FRAMES)
        medium_shots, medium_prompts = infer_with_retry(medium_frames, unified_sys)
        result["Medium_Shots"] = medium_shots
        result["Medium_Output"] = medium_prompts

        # Hard
        hard_frames = sample_frames_base64(sample["hard"]["video_path"], NUM_FRAMES)
        shots, hard_prompts = infer_with_retry(hard_frames, unified_sys)
        result["Shots_Output"] = shots
        result["Hard_Output"] = hard_prompts

    except FatalAPIError:
        raise
    except Exception as e:
        result.setdefault("Easy_Output", f"ERROR: {e}")
        result.setdefault("Medium_Output", f"ERROR: {e}")
        result.setdefault("Hard_Output", f"ERROR: {e}")

    return result


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    # ── Lock on local /tmp (NFS makes fcntl unreliable on /p/) ──
    lock_path = "/tmp/doubao_seed_inference.lock"
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another instance is already running. Exiting.")
        sys.exit(1)
    lock_file.write(str(os.getpid()))
    lock_file.flush()

    with open(SPLIT_A_PATH, "r", encoding="utf-8") as f:
        split_a = json.load(f)
    unified_sys = load_system_prompt(UNIFIED_PROMPT_PATH)

    # Load existing clean results: pick the source with more clean entries
    best_results = []
    for src in [OUTPUT_PATH, BACKUP_PATH]:
        if src and os.path.exists(src):
            try:
                with open(src, "r", encoding="utf-8") as f:
                    data = json.load(f)
                clean = [r for r in data if is_clean(r)]
                if len(clean) > len(best_results):
                    best_results = clean
                    print(f"  [{src}] has {len(clean)} clean / {len(data)} total")
            except Exception as e:
                print(f"  [{src}] read failed: {e}")
    results = best_results
    completed_ids = {r["id"] for r in results}
    print(f"Loaded {len(results)} clean results, resuming...")

    file_lock = threading.Lock()
    results_lock = threading.Lock()
    total = len(split_a)
    pending = [s for s in split_a if s["id"] not in completed_ids]
    print(f"Pending: {len(pending)} samples | Workers: {NUM_WORKERS}")

    fatal_error = None

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_sample, s, unified_sys): s for s in pending}

        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
                with results_lock:
                    results.append(result)
                    completed_ids.add(result["id"])
                merged = save_results(results, OUTPUT_PATH, file_lock)
                clean_count = sum(1 for r in merged if is_clean(r))
                print(f"  [ID {result['id']}] Saved. Clean: {clean_count}/{total}")

            except FatalAPIError as e:
                fatal_error = e
                # Cancel all pending (not-yet-started) futures
                for f in futures:
                    f.cancel()
                break

    # Final save with all clean results collected so far
    merged = save_results(results, OUTPUT_PATH, file_lock)
    clean_count = sum(1 for r in merged if is_clean(r))

    if fatal_error:
        print(f"\n[FATAL] {fatal_error}")
        print(f"Saved {clean_count} clean results. Rerun after recharging account.")
        sys.exit(1)

    print(f"\nDone! {clean_count} clean results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
