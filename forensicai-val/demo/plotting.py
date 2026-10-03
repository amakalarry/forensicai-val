"""
Real figure generation for the demo packages.

The figures in these packages used to be placeholder bytes, because the checker
only hashes a figure and reads its provenance sidecar -- it never looks at
pixels. That was fine for the test and wrong for a public repository: a file
named .png should be a .png. These are now genuine plots derived from the
package's own predictions and results.csv.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tool"))
from faval import iou, average_precision  # noqa: E402

INK = "#1F3A5F"
WARM = "#B4552D"
GRID = "#D8DEE6"


def _pr_points(gt_anns, pred_dets, cls, thr=0.5):
    gts = [g for g in gt_anns if g["category_id"] == cls]
    dts = sorted([d for d in pred_dets if d["category_id"] == cls], key=lambda d: -d["score"])
    if not gts:
        return [], []
    by_img = {}
    for g in gts:
        by_img.setdefault(g["image_id"], []).append(g["bbox"])
    matched = {k: [False] * len(v) for k, v in by_img.items()}
    tp, fp = [], []
    for d in dts:
        cands = by_img.get(d["image_id"], [])
        best_i, best_iou = -1, 0.0
        for i, gbox in enumerate(cands):
            if matched[d["image_id"]][i]:
                continue
            v = iou(d["bbox"], gbox)
            if v > best_iou:
                best_iou, best_i = v, i
        if best_iou >= thr and best_i >= 0:
            matched[d["image_id"]][best_i] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    c_tp = c_fp = 0
    recalls, precisions = [], []
    for t, f in zip(tp, fp):
        c_tp += t; c_fp += f
        recalls.append(c_tp / len(gts))
        precisions.append(c_tp / (c_tp + c_fp))
    return recalls, precisions


def per_class_ap(gt_anns, pred_dets, n_classes, thr=0.5):
    """Average precision for each class, computed the same way the checker does."""
    out = []
    for cls in range(n_classes):
        r, p = _pr_points(gt_anns, pred_dets, cls, thr)
        out.append(average_precision(r, p) if r else 0.0)
    return out


def _frame(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors="#444", labelsize=8)


def plot_class_performance(aps, classes, path, run_id, mean_label=True):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    order = np.argsort(aps)[::-1]
    names = [classes[i] for i in order]
    vals = [aps[i] for i in order]
    ax.bar(range(len(vals)), vals, color=INK, width=0.68)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylabel("AP@0.50")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if mean_label:
        m = float(np.mean(aps))
        ax.axhline(m, color=WARM, lw=1.2, ls="--")
        ax.text(len(vals) - 0.4, m + 0.018, f"mAP@50 = {m:.4f}",
                color=WARM, fontsize=8, ha="right")
    ax.set_title(f"Per-class detection performance  ({run_id})",
                 color=INK, fontsize=10.5, pad=10, loc="left")
    _frame(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pr_curve(gt_anns, pred_dets, classes, path, run_id):
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=130)
    for cls in range(len(classes)):
        r, p = _pr_points(gt_anns, pred_dets, cls)
        if r:
            ax.plot(r, p, lw=0.9, alpha=0.45, color=INK)
    # mean curve on a common recall grid
    grid = np.linspace(0, 1, 101)
    stack = []
    for cls in range(len(classes)):
        r, p = _pr_points(gt_anns, pred_dets, cls)
        if r:
            stack.append(np.interp(grid, r, p, left=p[0], right=0.0))
    if stack:
        ax.plot(grid, np.mean(stack, axis=0), lw=2.2, color=WARM, label="mean")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Precision-recall, 13 classes  ({run_id})",
                 color=INK, fontsize=10.5, pad=10, loc="left")
    _frame(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(gt_anns, pred_dets, classes, path, run_id):
    n = len(classes)
    m = np.zeros((n, n))
    # match detections to ground truth per image, count class agreement
    by_img = {}
    for g in gt_anns:
        by_img.setdefault(g["image_id"], []).append(g)
    used = {k: [False] * len(v) for k, v in by_img.items()}
    for d in sorted(pred_dets, key=lambda d: -d["score"]):
        cands = by_img.get(d["image_id"], [])
        best_i, best_iou = -1, 0.0
        for i, g in enumerate(cands):
            if used[d["image_id"]][i]:
                continue
            v = iou(d["bbox"], g["bbox"])
            if v > best_iou:
                best_iou, best_i = v, i
        if best_iou >= 0.5 and best_i >= 0:
            used[d["image_id"]][best_i] = True
            m[cands[best_i]["category_id"], d["category_id"]] += 1
    row = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, row, out=np.zeros_like(m), where=row > 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=130)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=38, ha="right", fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix, row-normalised  ({run_id})",
                 color=INK, fontsize=10.5, pad=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def read_results_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))
