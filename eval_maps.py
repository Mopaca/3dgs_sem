#
# Test-view error-map statistics for the decisive experiment ("guided by X, evaluated by Y").
#
# For every method folder under <model>/test/ (e.g. ours_30000) and every test view, this computes
# the DINOv2 cosine, SSIM and 1-L1 goodness maps between the render and the GT and reports
# lower-tail statistics (mean, 5th / 10th percentile, fraction of pixels below fixed thresholds).
# Results are written to <model>/eval_maps.json.
#
# Usage (run from the repository root so that torch.hub finds dinov2/):
#   python eval_maps.py -m output/decisive/flowers/dino_l03/seed0 [-m ...]
#   python eval_maps.py -m <model> --signals ssim l1          # without DINOv2
#   python eval_maps.py -m <model> --save_maps 3              # also save maps of the first 3 views (.npy)
#

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.signal_maps import GuidanceMaps, SIGNALS

# Thresholds on the goodness maps; "frac_below_t" = fraction of pixels with goodness < t.
THRESHOLDS = {
    "dino": [0.5, 0.7],
    "ssim": [0.5, 0.7],
    "l1": [0.90, 0.95],       # i.e. absolute error > 0.10 / > 0.05
    "l1_blur": [0.90, 0.95],
}


def load_image(path, device):
    """[3, H, W] float in [0, 1] (same values as torchvision's to_tensor for 8-bit PNGs)."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


@torch.no_grad()
def map_stats(gmap, thresholds):
    v = gmap.flatten().float()
    out = {"mean": v.mean().item(),
           "p05": torch.quantile(v, 0.05).item(),
           "p10": torch.quantile(v, 0.10).item()}
    for t in thresholds:
        out[f"frac_below_{t:g}"] = (v < t).float().mean().item()
    return out


def evaluate_model(model_path, guidance, signals, save_maps, device):
    test_dir = Path(model_path) / "test"
    if not test_dir.is_dir():
        print(f"[eval_maps] no test/ folder in {model_path}, skipped")
        return None

    result = {}
    for method in sorted(os.listdir(test_dir)):
        renders_dir = test_dir / method / "renders"
        gt_dir = test_dir / method / "gt"
        if not renders_dir.is_dir() or not gt_dir.is_dir():
            continue
        names = sorted(os.listdir(renders_dir))
        per_view = {s: {} for s in signals}
        for i, name in enumerate(names):
            render = load_image(renders_dir / name, device)
            gt = load_image(gt_dir / name, device)
            for s in signals:
                gmap = guidance.goodness(s, render, gt)  # [1, H, W]
                per_view[s][name] = map_stats(gmap, THRESHOLDS[s])
                if i < save_maps:
                    out_dir = test_dir / method / "guidance_maps"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / f"{Path(name).stem}_{s}.npy", gmap.squeeze(0).half().cpu().numpy())

        summary = {}
        for s in signals:
            keys = next(iter(per_view[s].values())).keys() if per_view[s] else []
            summary[s] = {k: float(np.mean([v[k] for v in per_view[s].values()])) for k in keys}
        result[method] = {"summary": summary, "per_view": per_view, "num_views": len(names)}
        print(f"[eval_maps] {model_path} / {method}: " +
              ", ".join(f"{s} mean={summary[s].get('mean', float('nan')):.4f} p05={summary[s].get('p05', float('nan')):.4f}"
                        for s in signals))

    with open(Path(model_path) / "eval_maps.json", "w") as f:
        json.dump({"signals": signals, "thresholds": {s: THRESHOLDS[s] for s in signals}, "methods": result}, f, indent=1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Guidance-map statistics on rendered test views")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str)
    parser.add_argument("--signals", nargs="+", default=["dino", "ssim", "l1"], choices=list(SIGNALS))
    parser.add_argument("--save_maps", type=int, default=0, help="save maps of the first N views as .npy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_paths = [os.path.abspath(p) for p in args.model_paths]
    # torch.hub.load("dinov2/", source="local") is resolved relative to the working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    guidance = GuidanceMaps(args.signals, device=device)
    for mp in model_paths:
        evaluate_model(mp, guidance, args.signals, args.save_maps, device)


if __name__ == "__main__":
    main()
