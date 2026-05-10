"""
Compare pretrained vs. fine-tuned DiffSplat on the validation character.

Usage (inside conda env diffsplat_sm120):
    PYTHONPATH=. python eval_metrics.py \
        --pred_dir out/gsdiff_sd15_chibi_image_run/inference \
        --pred_iter 500 \
        --baseline_dir out/gsdiff_gobj83k_sd15_image__render/inference \
        --baseline_iter 13020 \
        --val_parquet data/chibi_train/val.parquet \
        --out_png metrics_plot.png
"""

import argparse
import io
import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.utils.metrics import ImageConditionMetrics


# Azimuths DiffSplat generates: 0, 90, 180, 270 deg → view indices in parquet
# Blender renders 40 views evenly spaced; 360/40 = 9 deg per step
# 0° → view 00, 90° → view 10, 180° → view 20, 270° → view 30
EVAL_VIEW_KEYS = ["00000.png", "00010.png", "00020.png", "00030.png"]


def load_gt_views(val_parquet: str) -> list:
    df = pd.read_parquet(val_parquet)
    row = df.iloc[0]
    images = []
    for k in EVAL_VIEW_KEYS:
        img = Image.open(io.BytesIO(row[k])).convert("RGB")
        images.append(img)
    return images


def load_pred_strip(pred_dir: str, pred_iter: int) -> list:
    """Load the 4-view horizontal strip PNG saved by infer_gsdiff_sd.py and split it."""
    pattern = os.path.join(pred_dir, f"*_{pred_iter:06d}_gs.png")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern}")
    strip = np.array(Image.open(matches[-1]).convert("RGB"))  # (H, 4W, 3)
    W = strip.shape[1] // 4
    return [Image.fromarray(strip[:, i*W:(i+1)*W]) for i in range(4)]


def compute_and_print(name: str, preds: list, gts: list, metrics: ImageConditionMetrics):
    psnr, ssim, lpips = metrics.evaluate(preds, gts)
    print(f"{name}:")
    print(f"  PSNR  = {psnr[0]:.3f} ± {psnr[1]:.3f}")
    print(f"  SSIM  = {ssim[0]:.4f} ± {ssim[1]:.4f}")
    print(f"  LPIPS = {lpips[0]:.4f} ± {lpips[1]:.4f}")
    return psnr[0], ssim[0], lpips[0]


def make_bar_chart(results: dict, out_png: str):
    metrics = ["PSNR ↑", "SSIM ↑", "LPIPS ↓"]
    keys    = list(results.keys())
    values  = {m: [results[k][i] for k in keys] for i, m in enumerate(metrics)}

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    colors = ["#3b82f6", "#10b981"]
    for ax, metric in zip(axes, metrics):
        bars = ax.bar(keys, values[metric], color=colors[:len(keys)], width=0.5)
        for bar, v in zip(bars, values[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(metric, fontsize=13)
        ax.set_ylim(0, max(values[metric]) * 1.25)
        ax.tick_params(axis="x", labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pretrained vs. Fine-tuned DiffSplat on Anime Val Character", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",      required=True)
    parser.add_argument("--pred_iter",     type=int, required=True)
    parser.add_argument("--baseline_dir",  default=None,
                        help="inference dir of pretrained model (optional)")
    parser.add_argument("--baseline_iter", type=int, default=13020)
    parser.add_argument("--val_parquet",   default="data/chibi_train/val.parquet")
    parser.add_argument("--out_png",       default="metrics_plot.png")
    parser.add_argument("--gpu",           type=int, default=0)
    args = parser.parse_args()

    metrics = ImageConditionMetrics(lpips_net="vgg", lpips_res=256, device_idx=args.gpu)
    gt_views = load_gt_views(args.val_parquet)

    results = {}

    if args.baseline_dir:
        baseline_preds = load_pred_strip(args.baseline_dir, args.baseline_iter)
        p, s, l = compute_and_print("Pretrained", baseline_preds, gt_views, metrics)
        results["Pretrained"] = (p, s, l)

    ft_preds = load_pred_strip(args.pred_dir, args.pred_iter)
    p, s, l  = compute_and_print("Fine-tuned", ft_preds, gt_views, metrics)
    results["Fine-tuned"] = (p, s, l)

    if len(results) >= 1:
        make_bar_chart(results, args.out_png)


if __name__ == "__main__":
    main()
