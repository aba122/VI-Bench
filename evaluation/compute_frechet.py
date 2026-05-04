"""
Standalone Fréchet distance computation for Prompt Inversion Benchmark.

Reads video paths from results/Video/*_Video.json, extracts InceptionV3 (FID)
and I3D (FVD) embeddings, then computes distribution-level Fréchet distances.

GT embeddings are computed once and shared across all models.

Usage:
  conda activate Hunyuan
  python evaluate/compute_frechet.py
"""

import os, json, glob
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
from datetime import datetime

try:
    from decord import VideoReader, cpu as d_cpu
    DECORD_OK = True
except ImportError:
    DECORD_OK = False
    import cv2

# ─── Config ──────────────────────────────────────────────────────────────────
INCEPTION_CACHE = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth"
)
I3D_PATH = "/bigtemp/fzv6en/xwl/Prompt_Inversion_Bench/model/i3d_torchscript.pt"

N_FRAMES_FID   = 8
N_FRAMES_FVD   = 16
I3D_SIZE       = 224
INCEPTION_SIZE = 299

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_A_PATH = os.path.join(BASE_DIR, "Dataset", "Videos", "Split_A.json")
VIDEO_DIR    = os.path.join(BASE_DIR, "results", "Video")
LOG_PATH     = os.path.join(BASE_DIR, "Experiment.log")


# ─── Models ──────────────────────────────────────────────────────────────────
class _InceptionV3Pool3(torch.nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        net = tv_models.inception_v3(
            pretrained=False, transform_input=False, aux_logits=True
        )
        state = torch.load(weights_path, map_location="cpu")
        net.load_state_dict(state)
        self.net = net
        self._feat = None
        self.net.avgpool.register_forward_hook(
            lambda m, i, o: setattr(self, "_feat", o.flatten(1))
        )

    @torch.no_grad()
    def forward(self, x):
        self._feat = None
        self.net(x)
        return self._feat


def load_inception(device):
    return _InceptionV3Pool3(INCEPTION_CACHE).to(device).eval()


def load_i3d(device):
    return torch.jit.load(I3D_PATH, map_location=device).eval()


# ─── Frame loading ────────────────────────────────────────────────────────────
def load_frames(video_path, n_frames):
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        if DECORD_OK:
            vr = VideoReader(video_path, ctx=d_cpu(0))
            idx = np.linspace(0, len(vr) - 1, n_frames, dtype=int)
            return [Image.fromarray(f) for f in vr.get_batch(idx).asnumpy()]
        else:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = set(np.linspace(0, total - 1, n_frames, dtype=int).tolist())
            frames, fi = [], 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if fi in idx:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                fi += 1
            cap.release()
            return frames if frames else None
    except Exception:
        return None


def load_frames_multi(video_paths, n_frames):
    """Sample n_frames proportionally across multiple videos."""
    valid = [p for p in video_paths if p and os.path.exists(p)]
    if not valid:
        return None
    counts = []
    for p in valid:
        try:
            if DECORD_OK:
                counts.append(len(VideoReader(p, ctx=d_cpu(0))))
            else:
                cap = cv2.VideoCapture(p)
                counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                cap.release()
        except Exception:
            counts.append(0)
    total_fc = sum(counts)
    if total_fc == 0:
        return None
    budgets = [max(1, round(n_frames * c / total_fc)) for c in counts]
    all_frames = []
    for p, n in zip(valid, budgets):
        fs = load_frames(p, n)
        if fs:
            all_frames.extend(fs)
    if not all_frames:
        return None
    if len(all_frames) != n_frames:
        idx = np.linspace(0, len(all_frames) - 1, n_frames, dtype=int)
        all_frames = [all_frames[i] for i in idx]
    return all_frames


# ─── Embedding extraction ─────────────────────────────────────────────────────
_fid_tf = T.Compose([T.Resize(INCEPTION_SIZE), T.CenterCrop(INCEPTION_SIZE), T.ToTensor()])
_fvd_tf = T.Compose([T.Resize(I3D_SIZE),       T.CenterCrop(I3D_SIZE),       T.ToTensor()])


def get_fid_emb(video_path, inception, device):
    frames = load_frames(video_path, N_FRAMES_FID)
    if not frames:
        return None
    imgs = torch.stack([_fid_tf(f) for f in frames]).to(device)
    feats = inception(imgs).mean(0)
    norm = feats.norm()
    return (feats / norm).cpu().float().numpy() if norm > 1e-8 else None


def get_fvd_emb(frames, i3d, device):
    """Encode a list of PIL frames into I3D embedding."""
    if not frames:
        return None
    t = torch.stack([_fvd_tf(f) for f in frames]) * 2.0 - 1.0
    clip = t.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = i3d(clip)[0]
    norm = feats.norm()
    return (feats / norm).cpu().float().numpy() if norm > 1e-8 else None


def get_fvd_emb_path(video_path, i3d, device):
    return get_fvd_emb(load_frames(video_path, N_FRAMES_FVD), i3d, device)


def get_fvd_emb_multi(video_paths, i3d, device):
    return get_fvd_emb(load_frames_multi(video_paths, N_FRAMES_FVD), i3d, device)


# ─── Fréchet distance ─────────────────────────────────────────────────────────
def frechet(embs_a, embs_b, eps=1e-6):
    a = np.stack([e for e in embs_a if e is not None])
    b = np.stack([e for e in embs_b if e is not None])
    if len(a) < 2 or len(b) < 2:
        return None
    mu1, s1 = a.mean(0), np.cov(a, rowvar=False)
    mu2, s2 = b.mean(0), np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((s1 + np.eye(s1.shape[0]) * eps) @
                                (s2 + np.eye(s2.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading InceptionV3 ...")
    inception = load_inception(device)
    print("Loading I3D ...")
    i3d = load_i3d(device)
    print("Models loaded.\n")

    with open(SPLIT_A_PATH) as f:
        gt_list = json.load(f)
    gt_by_id = {x["id"]: x for x in gt_list}

    # ── Pre-compute GT embeddings (shared across all models) ──
    print("Extracting GT embeddings ...")
    gt_fid_easy, gt_fid_medium, gt_fid_hard = [], [], []
    gt_fvd_easy, gt_fvd_medium, gt_fvd_hard = [], [], []

    for i, gt in enumerate(gt_list):
        if (i + 1) % 50 == 0:
            print(f"  GT {i+1}/{len(gt_list)}")
        e_path = gt["easy"]["video_path"]
        m_path = gt["medium"]["video_path"]
        h_path = gt["hard"].get("video_path")

        fid_e = get_fid_emb(e_path, inception, device)
        fid_m = get_fid_emb(m_path, inception, device)
        fid_h = get_fid_emb(h_path, inception, device)
        fvd_e = get_fvd_emb_path(e_path, i3d, device)
        fvd_m = get_fvd_emb_path(m_path, i3d, device)
        fvd_h = get_fvd_emb_path(h_path, i3d, device)

        if fid_e is not None: gt_fid_easy.append(fid_e)
        if fid_m is not None: gt_fid_medium.append(fid_m)
        if fid_h is not None: gt_fid_hard.append(fid_h)
        if fvd_e is not None: gt_fvd_easy.append(fvd_e)
        if fvd_m is not None: gt_fvd_medium.append(fvd_m)
        if fvd_h is not None: gt_fvd_hard.append(fvd_h)

    print(f"  GT done: easy={len(gt_fid_easy)} medium={len(gt_fid_medium)} hard={len(gt_fid_hard)}\n")

    # ── Per-model pred embeddings ──
    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*_Video.json")))
    all_results = []

    for vf in video_files:
        model_name = os.path.basename(vf).replace("_Video.json", "")
        print(f"{'='*60}\n{model_name}")
        with open(vf) as f:
            data = json.load(f)

        pred_fid_easy, pred_fid_medium, pred_fid_hard = [], [], []
        pred_fvd_easy, pred_fvd_medium, pred_fvd_hard = [], [], []

        for i, item in enumerate(data):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(data)}")
            gt = gt_by_id.get(item["id"])
            if gt is None:
                continue

            # Easy
            e = get_fid_emb(item.get("Easy_Video"), inception, device)
            if e is not None: pred_fid_easy.append(e)
            e = get_fvd_emb_path(item.get("Easy_Video"), i3d, device)
            if e is not None: pred_fvd_easy.append(e)

            # Medium
            m = get_fid_emb(item.get("Medium_Video"), inception, device)
            if m is not None: pred_fid_medium.append(m)
            m = get_fvd_emb_path(item.get("Medium_Video"), i3d, device)
            if m is not None: pred_fvd_medium.append(m)

            # Hard (concat shots for FVD)
            shot_keys = sorted(k for k in item if k.startswith("Hard_Video_shot"))
            shot_paths = [item.get(k) for k in shot_keys]

            # FID hard: mean of per-shot embeddings
            shot_fid_embs = [get_fid_emb(p, inception, device) for p in shot_paths if p]
            shot_fid_embs = [e for e in shot_fid_embs if e is not None]
            if shot_fid_embs:
                h_fid = np.mean(shot_fid_embs, axis=0)
                h_fid /= np.linalg.norm(h_fid) + 1e-8
                pred_fid_hard.append(h_fid)

            # FVD hard: concat all shots
            h_fvd = get_fvd_emb_multi(shot_paths, i3d, device)
            if h_fvd is not None:
                pred_fvd_hard.append(h_fvd)

        # Fréchet distances
        res = {
            "model":      model_name,
            "FID_easy":   frechet(pred_fid_easy,   gt_fid_easy),
            "FID_medium": frechet(pred_fid_medium, gt_fid_medium),
            "FID_hard":   frechet(pred_fid_hard,   gt_fid_hard),
            "FVD_easy":   frechet(pred_fvd_easy,   gt_fvd_easy),
            "FVD_medium": frechet(pred_fvd_medium, gt_fvd_medium),
            "FVD_hard":   frechet(pred_fvd_hard,   gt_fvd_hard),
            "n_easy":     len(pred_fid_easy),
            "n_medium":   len(pred_fid_medium),
            "n_hard":     len(pred_fid_hard),
        }
        def _f(v): return f"{v:.4f}" if v is not None else "N/A"
        print(f"  FID  easy={_f(res['FID_easy'])}  med={_f(res['FID_medium'])}  hard={_f(res['FID_hard'])}")
        print(f"  FVD  easy={_f(res['FVD_easy'])}  med={_f(res['FVD_medium'])}  hard={_f(res['FVD_hard'])}")
        all_results.append(res)

    # ── Rankings ──
    def _avg(r, keys):
        vals = [r[k] for k in keys if r[k] is not None]
        return sum(vals) / len(vals) if vals else None

    for r in all_results:
        r["FID_overall"] = _avg(r, ["FID_easy", "FID_medium", "FID_hard"])
        r["FVD_overall"] = _avg(r, ["FVD_easy", "FVD_medium", "FVD_hard"])

    def print_ranking(title, key_overall, keys, note="lower=better"):
        print(f"\n{'='*70}")
        print(f"{title}  ({note})")
        print(f"{'='*70}")
        rows = [(r, r[key_overall]) for r in all_results if r[key_overall] is not None]
        rows.sort(key=lambda x: x[1])  # lower is better for Fréchet
        hdr = f"  {'Model':<20} {'Easy':>10} {'Medium':>10} {'Hard':>10} {'Overall':>10}"
        print(hdr)
        print("  " + "-" * 64)
        for i, (r, _) in enumerate(rows):
            vals = [r[k] for k in keys]
            def _fmt(v): return f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
            print(f"  {r['model']:<20}" + "".join(_fmt(v) for v in vals) +
                  f"{_fmt(r[key_overall])}")

    print_ranking("FID  (InceptionV3 2048-dim, Fréchet distance)", "FID_overall",
                  ["FID_easy", "FID_medium", "FID_hard"])
    print_ranking("FVD  (I3D Kinetics-400 400-dim, Fréchet distance)", "FVD_overall",
                  ["FVD_easy", "FVD_medium", "FVD_hard"])

    # ── Append to Experiment.log ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr_line = (f"  {'Model':<20} {'FID Easy':>10} {'FID Med':>10} {'FID Hard':>10} "
                f"{'FID Ovrl':>10} | {'FVD Easy':>10} {'FVD Med':>10} "
                f"{'FVD Hard':>10} {'FVD Ovrl':>10}")
    sep_line = "  " + "-" * 100

    def _lf(v): return f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"

    log_rows = []
    for r in sorted(all_results, key=lambda x: (x["FVD_overall"] or 999)):
        log_rows.append(
            f"  {r['model']:<20}"
            f"{_lf(r['FID_easy'])}{_lf(r['FID_medium'])}{_lf(r['FID_hard'])}{_lf(r['FID_overall'])} |"
            f"{_lf(r['FVD_easy'])}{_lf(r['FVD_medium'])}{_lf(r['FVD_hard'])}{_lf(r['FVD_overall'])}"
        )

    log_block = "\n".join([
        f"\n[{timestamp}] Fréchet Distance Rankings  "
        f"(InceptionV3 2048-dim FID, I3D 400-dim FVD, lower=better)",
        f"  GT embeddings: {len(gt_fid_easy)} easy / {len(gt_fid_medium)} medium / {len(gt_fid_hard)} hard",
        hdr_line, sep_line, *log_rows, "",
    ])
    with open(LOG_PATH, "a") as f:
        f.write(log_block)
    print(f"\nResults appended to {LOG_PATH}")


if __name__ == "__main__":
    main()
