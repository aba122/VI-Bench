"""
LLM-Judge evaluation for Prompt Inversion Benchmark.
Judge model: gpt-4o-2024-11-20 via 302.ai API.

Unified system prompt for all three difficulty levels.

Scoring (all levels use same 5-dimension formula):
  score = (subject + action + scene + style + camera - 5) / 10  → [0, 1]

  Critical rule applied by judge:
    If the GT does not explicitly mention style or camera, those dimensions
    are auto-scored 3 (not required → no penalty).
    This structurally ensures Easy ≥ Medium in expectation, because Easy GT
    contains no style/camera terms while Medium GT does.

  Easy  — GT is rewritten narrative (~100 words), no style/camera terms
           → style/camera auto-3 → score ceiling is high
  Medium — GT includes style, aesthetic, camera descriptions
           → all 5 dims must be captured → harder
  Hard  — per-shot evaluation using same system prompt (sequential alignment)
           final_score = mean(shot_scores for matched pairs)

Hard alignment rule:
  n = GT shot count, m = pred shot count, k = min(n, m)
  - Positions 1..k : evaluate pair independently
  - Extra pred shots (k+1..m) : not evaluated, not penalized
  - Missing pred shots (k+1..n) : score 0 (denominator = n, not k)
  - final_score = sum(k evaluated scores) / n

Overall = (Easy_score + Medium_score + Hard_final_score) / 3

Resume: if a sample already has all three LLM-Judge keys, it is skipped.
Output is saved after every sample.

Usage:
  export API_302_KEY='your-key'
  python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/evaluate/LLM-Judge.py
"""

import os
import json
import glob
import http.client
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────
API_KEY       = os.getenv("API_302_KEY", "sk-kVZOxtEdQOhIxKSxSvrUpXVh6ydG744GPqfnnxwfxtwaZa8U")
API_HOST      = "api.302.ai"
MODEL         = "gpt-4o-2024-11-20"
MAX_RETRIES   = 5
RETRY_DELAY   = 3      # base seconds, exponential backoff
SLEEP_BETWEEN = 0.5    # seconds between API calls

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR    = os.path.join(BASE_DIR, "results", "Video")
OUTPUT_DIR   = os.path.join(BASE_DIR, "results", "LLM-Judge")
SPLIT_A_PATH = os.path.join(BASE_DIR, "Dataset", "Videos", "Split_A.json")
LOG_PATH     = os.path.join(BASE_DIR, "Experiment.log")


# ─── Unified System Prompt ───────────────────────────────────────────────────
UNIFIED_SYSTEM = """\
You are evaluating a Prompt Inversion task for AI-generated videos.

The model watched an AI-generated video and predicted the original text-to-video \
generation prompt. Your task: score how well the Prediction captures what the \
Ground Truth (GT) describes.

Evaluate these five dimensions. Assign 1–5 for each:
  5 = essentially identical
  4 = mostly correct, only minor differences
  3 = partially correct, notable differences
  2 = superficially similar, mostly wrong
  1 = completely wrong or missing

Dimensions:
  subject (1-5): Is the main subject/entity in the GT correctly identified?
    5 = subject type, appearance and role essentially identical
    4 = subject type correct, minor appearance/role differences
    3 = subject type correct but notable differences in appearance or behavior
    2 = related but different subject type (e.g. wrong animal species)
    1 = completely different subject
  action (1-5): Is the core action or event correctly described?
    5 = action essentially identical
    4 = main action matches, minor detail differences
    3 = general direction of action similar but notable differences
    2 = only superficial action similarity
    1 = completely different action
  scene (1-5): Is the scene/setting/context correctly reflected?
    5 = location, environment and atmosphere essentially identical
    4 = location type matches, minor atmospheric differences
    3 = location type matches but atmosphere/details differ notably
    2 = broadly similar but key environmental features differ
    1 = completely different scene
  style (1-5): Are visual style, aesthetic, or atmosphere elements captured?
    5 = color palette, mood and aesthetic essentially identical
    4 = overall visual tone matches, minor style differences
    3 = general mood similar but color/texture differs notably
    2 = broadly similar aesthetic but overall feel differs
    1 = completely different visual style
  camera (1-5): Are camera angle, movement, or framing elements captured?
    5 = shot type, angle and movement essentially identical
    4 = shot type matches, minor camera differences
    3 = shot type similar but camera angle/movement differs notably
    2 = either shot type or angle notably different
    1 = completely different cinematographic approach

Critical rule: If the GT does not explicitly mention a dimension (e.g., contains no \
style description or no camera/framing description), assign 5 for that dimension \
automatically — it is not required, so no penalty applies.

Output valid JSON only, no markdown:
{"subject": <1-5>, "action": <1-5>, "scene": <1-5>, "style": <1-5>, "camera": <1-5>}\
"""


# ─── API Call ─────────────────────────────────────────────────────────────────
def call_judge(system_prompt, user_content):
    """Call gpt-4o-2024-11-20 via 302.ai. Returns parsed dict."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    })

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        conn = None
        try:
            conn = http.client.HTTPSConnection(API_HOST, timeout=60)
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res  = conn.getresponse()
            raw  = res.read().decode("utf-8")

            if res.status != 200:
                raise RuntimeError(f"HTTP {res.status}: {raw[:300]}")

            content = json.loads(raw)["choices"][0]["message"]["content"].strip()

            # Tolerant JSON extraction (handles markdown code blocks)
            match = re.search(r'\{[\s\S]*\}', content)
            if not match:
                raise ValueError(f"No JSON in response: {content[:200]}")

            return json.loads(match.group())

        except Exception as e:
            last_err = str(e)
            print(f"      [Attempt {attempt}/{MAX_RETRIES}] {e}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))

    raise RuntimeError(f"Judge failed after {MAX_RETRIES} retries. Last: {last_err}")


# ─── Scoring ─────────────────────────────────────────────────────────────────
def score_unified(r):
    """Unified 5-dimension formula for all levels. Each dim 1-5 → total [5,25] → score [0,1]."""
    return round(
        (r["subject"] + r["action"] + r["scene"] + r["style"] + r["camera"] - 5) / 20,
        6,
    )

def is_valid(text):
    return isinstance(text, str) and text.strip() and not text.upper().startswith("ERROR")

def extract_pred(raw):
    """Extract prediction string from either a plain string or a list of shots."""
    if isinstance(raw, list):
        return raw[0] if raw and isinstance(raw[0], str) else None
    return raw


# ─── Level Evaluators ────────────────────────────────────────────────────────
def evaluate_easy(pred, gt_prompt):
    user_msg = f"[Ground Truth Prompt]\n{gt_prompt}\n\n[Predicted Prompt]\n{pred}"
    result = call_judge(UNIFIED_SYSTEM, user_msg)
    time.sleep(SLEEP_BETWEEN)
    result["score"] = score_unified(result)
    return result


def evaluate_medium(pred, gt_prompt):
    user_msg = f"[Ground Truth Prompt]\n{gt_prompt}\n\n[Predicted Prompt]\n{pred}"
    result = call_judge(UNIFIED_SYSTEM, user_msg)
    time.sleep(SLEEP_BETWEEN)
    result["score"] = score_unified(result)
    return result


def evaluate_hard(pred_shots, gt_shot_texts):
    """
    Sequential alignment:
      k = min(n, m)
      Positions 1..k   → evaluate pair independently (using UNIFIED_SYSTEM)
      Extra pred shots (k+1..m) → not evaluated, not penalized
      Missing shots (k+1..n)   → score 0; denominator = n (gt_shots)
      final_score = sum(k evaluated scores) / n
    """
    n = len(gt_shot_texts)
    m = len(pred_shots)
    k = min(n, m)
    shot_results = []

    for i in range(k):
        shot_num  = i + 1
        gt_shot   = gt_shot_texts[i]
        pred_shot = pred_shots[i]

        if not is_valid(pred_shot):
            shot_results.append({"shot": shot_num, "invalid_pred": True, "score": None})
            continue

        user_msg = (
            f"[Ground Truth Shot {shot_num}]\n{gt_shot}\n\n"
            f"[Predicted Shot {shot_num}]\n{pred_shot}"
        )
        try:
            r = call_judge(UNIFIED_SYSTEM, user_msg)
            time.sleep(SLEEP_BETWEEN)
            r["shot"]  = shot_num
            r["score"] = score_unified(r)
            shot_results.append(r)
        except Exception as e:
            print(f"      Hard shot {shot_num} failed: {e}")
            shot_results.append({"shot": shot_num, "error": str(e), "score": None})

    valid_scores = [r["score"] for r in shot_results if r.get("score") is not None]
    # Denominator = gt_shots (n), so missed shots count as 0
    final_score  = sum(valid_scores) / n if n > 0 else 0.0
    return {
        "shot_results": shot_results,
        "gt_shots":     n,
        "pred_shots":   m,
        "matched_shots": k,
        "final_score":  round(final_score, 6),
    }


# ─── Per-file Processing ─────────────────────────────────────────────────────
def process_file(json_path, gt_by_id, n_workers=8):
    basename   = os.path.basename(json_path)
    model_name = basename.replace("_Video.json", "").replace(".json", "")
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}_LLM-Judge.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load or initialise output
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            output_list = json.load(f)
        output_by_id = {item["id"]: item for item in output_list}
        print(f"  Resumed: {len(output_list)} items already saved")
    else:
        output_by_id = {item["id"]: dict(item) for item in data}

    save_lock = threading.Lock()

    def save():
        ordered = [output_by_id[d["id"]] for d in data if d["id"] in output_by_id]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)

    total = len(data)
    completed = [0]  # mutable counter for progress

    def process_one(item, idx):
        item_id = item["id"]
        gt = gt_by_id.get(item_id)
        if gt is None:
            return

        with save_lock:
            out = output_by_id.get(item_id, dict(item))
            already_done = all(k in out for k in ("Easy_LLM_Judge", "Medium_LLM_Judge", "Hard_LLM_Judge"))

        if already_done:
            with save_lock:
                completed[0] += 1
                print(f"  [{completed[0]}/{total}] ID {item_id}: already done.")
            return

        # ── Easy ──
        if "Easy_LLM_Judge" not in out:
            pred = extract_pred(item.get("Easy_Output"))
            gt_p = gt["easy"]["prompt"]
            if is_valid(pred):
                try:
                    out["Easy_LLM_Judge"] = evaluate_easy(pred, gt_p)
                except Exception as e:
                    out["Easy_LLM_Judge"] = {"error": str(e), "score": None}
            else:
                out["Easy_LLM_Judge"] = {"invalid_pred": True, "score": None}

        # ── Medium ──
        if "Medium_LLM_Judge" not in out:
            pred = extract_pred(item.get("Medium_Output"))
            gt_p = gt["medium"]["rewrite_prompt"]
            if is_valid(pred):
                try:
                    out["Medium_LLM_Judge"] = evaluate_medium(pred, gt_p)
                except Exception as e:
                    out["Medium_LLM_Judge"] = {"error": str(e), "score": None}
            else:
                out["Medium_LLM_Judge"] = {"invalid_pred": True, "score": None}

        # ── Hard ──
        if "Hard_LLM_Judge" not in out:
            pred_hard      = item.get("Hard_Output")
            gt_hard_prompt = gt["hard"]["hard_prompt"]
            gt_shot_keys   = sorted(gt_hard_prompt.keys())
            gt_shot_texts  = [gt_hard_prompt[k] for k in gt_shot_keys]
            if isinstance(pred_hard, list) and len(pred_hard) > 0:
                try:
                    out["Hard_LLM_Judge"] = evaluate_hard(pred_hard, gt_shot_texts)
                except Exception as e:
                    out["Hard_LLM_Judge"] = {"error": str(e), "final_score": None}
            else:
                out["Hard_LLM_Judge"] = {"invalid_pred": True, "final_score": None}

        # ── Overall ──
        easy_s = (out.get("Easy_LLM_Judge")   or {}).get("score")
        med_s  = (out.get("Medium_LLM_Judge") or {}).get("score")
        hard_s = (out.get("Hard_LLM_Judge")   or {}).get("final_score")
        valid  = [s for s in (easy_s, med_s, hard_s) if s is not None]
        out["Overall_LLM_Judge"] = round(sum(valid) / len(valid), 6) if valid else None

        e_score = (out.get("Easy_LLM_Judge")   or {}).get("score")
        m_score = (out.get("Medium_LLM_Judge") or {}).get("score")
        h_score = (out.get("Hard_LLM_Judge")   or {}).get("final_score")

        with save_lock:
            output_by_id[item_id] = out
            completed[0] += 1
            cnt = completed[0]
            e_str = f"{e_score:.3f}" if e_score is not None else "n/a"
            m_str = f"{m_score:.3f}" if m_score is not None else "n/a"
            h_str = f"{h_score:.3f}" if h_score is not None else "n/a"
            print(f"  [{cnt}/{total}] ID {item_id}  E={e_str} M={m_str} H={h_str}", flush=True)
            save()

    print(f"  Processing {total} samples with {n_workers} parallel workers ...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_one, item, idx): idx for idx, item in enumerate(data)}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f"  Worker error: {exc}")

    save()
    print(f"  Saved → {output_path}")

    # Aggregate metrics
    final_list = [output_by_id[d["id"]] for d in data if d["id"] in output_by_id]

    def agg(key, subkey):
        vals = [
            o[key][subkey] for o in final_list
            if isinstance(o.get(key), dict) and o[key].get(subkey) is not None
        ]
        return (round(sum(vals) / len(vals), 4), len(vals)) if vals else (0.0, 0)

    easy_avg,  ne = agg("Easy_LLM_Judge",   "score")
    med_avg,   nm = agg("Medium_LLM_Judge", "score")
    hard_avg,  nh = agg("Hard_LLM_Judge",   "final_score")

    overall_vals = [o["Overall_LLM_Judge"] for o in final_list if o.get("Overall_LLM_Judge") is not None]
    overall_avg  = round(sum(overall_vals) / len(overall_vals), 4) if overall_vals else 0.0

    return {
        "model":           model_name,
        "Easy_LLM_Judge":  easy_avg,
        "Medium_LLM_Judge": med_avg,
        "Hard_LLM_Judge":  hard_avg,
        "Overall_LLM_Judge": overall_avg,
        "n_easy": ne, "n_medium": nm, "n_hard": nh,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None,
        help="Process only this model by name (e.g. Intern-VL25). Looks in VIDEO_DIR."
    )
    parser.add_argument(
        "--result-json", type=str, default=None,
        help="Path to a direct inference result JSON (e.g. results/Intern-VL25.json). "
             "Bypasses VIDEO_DIR lookup."
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel API workers (default: 8)."
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(SPLIT_A_PATH, "r", encoding="utf-8") as f:
        split_a = json.load(f)
    gt_by_id = {item["id"]: item for item in split_a}

    # ── Resolve which files to process ──
    if args.result_json:
        json_files = [args.result_json]
    else:
        all_video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_Video.json")))
        if args.model:
            json_files = [
                vf for vf in all_video_files
                if os.path.basename(vf) == f"{args.model}_Video.json"
            ]
            if not json_files:
                print(f"No file found for model '{args.model}'. Available:")
                for vf in all_video_files:
                    print(f"  {os.path.basename(vf).replace('_Video.json','')}")
                return
        else:
            json_files = all_video_files

    print(f"Processing {len(json_files)} file(s):")
    for jf in json_files:
        print(f"  {os.path.basename(jf)}")
    print()

    all_metrics = []

    for jf in json_files:
        model_name = os.path.basename(jf).replace("_Video.json", "").replace(".json", "")
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}  (workers={args.workers})")
        print(f"{'='*60}")
        metrics = process_file(jf, gt_by_id, n_workers=args.workers)
        all_metrics.append(metrics)

        print(f"\n  Easy   LLM-Judge: {metrics['Easy_LLM_Judge']:.4f}  (n={metrics['n_easy']})")
        print(f"  Medium LLM-Judge: {metrics['Medium_LLM_Judge']:.4f}  (n={metrics['n_medium']})")
        print(f"  Hard   LLM-Judge: {metrics['Hard_LLM_Judge']:.4f}  (n={metrics['n_hard']})")
        print(f"  Overall:          {metrics['Overall_LLM_Judge']:.4f}")

    # ── Experiment.log ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header    = f"{'Model':<20} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>8}"
    sep       = "-" * 56
    rows      = [
        f"{m['model']:<20} {m['Easy_LLM_Judge']:>8.4f} {m['Medium_LLM_Judge']:>8.4f} "
        f"{m['Hard_LLM_Judge']:>8.4f} {m['Overall_LLM_Judge']:>8.4f}"
        for m in all_metrics
    ]
    log_block = "\n".join([
        f"\n[{timestamp}] LLM-Judge Evaluation  judge={MODEL}",
        f"  Output: results/LLM-Judge/",
        header, sep, *rows, "",
    ])
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_block)
    print(f"\nResults appended to {LOG_PATH}")

    # ── Final table ──
    print(f"\n{'='*56}")
    print("LLM-Judge Final Summary")
    print(f"{'='*56}")
    print(header)
    print(sep)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

# Usage:
# export API_302_KEY='sk-...'
# python /p/fzv6enresearch/xwl/Prompt_Inversion_Bench/evaluate/LLM-Judge.py
