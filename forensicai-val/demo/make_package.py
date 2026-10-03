"""
Build a synthetic experiment package shaped like an Ultralytics YOLOv11 run
directory, with three seeded defects and several genuinely-consistent artifacts.

The package deliberately mixes correct and incorrect records so the checker has
to distinguish verified consistency, discrepancy, and insufficient evidence
rather than simply failing everything.

Seeded defects
  D1  figures/fig3_class_performance.png is a real plot of an EARLIER run
  D2  manuscript/reported_results.json overstates mAP@50 vs saved predictions
  D3  dataset/dataset.yaml carries no version, manifest hash, or image count
"""
import csv
import hashlib
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from plotting import (per_class_ap, plot_class_performance,
                      plot_confusion_matrix, plot_pr_curve)

ROOT = os.path.join(os.path.dirname(__file__), "experiment_package")
CURRENT_RUN = "run_2026-08-02_9c7d"
EARLIER_RUN = "run_2026-06-14_a3f1"

CLASSES = [
    "firearm", "knife", "blood_stain", "shell_casing", "fingerprint_lift",
    "footwear_impression", "narcotics_packet", "mobile_phone", "document",
    "glass_fragment", "fiber_sample", "tool_mark", "syringe",
]

rng = random.Random(20260929)
np.random.seed(20260929)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def make_detections(n_images=240):
    """Generate ground truth and predictions with a realistic, imperfect detector."""
    gt, preds = [], []
    for img_id in range(n_images):
        n_obj = rng.randint(1, 4)
        for _ in range(n_obj):
            cls = rng.randrange(len(CLASSES))
            x, y = rng.uniform(0, 500), rng.uniform(0, 400)
            w, h = rng.uniform(40, 140), rng.uniform(40, 140)
            gt.append({"image_id": img_id, "category_id": cls,
                       "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]})

            # detector recovers most objects, with jitter; misses some entirely
            if rng.random() < 0.86:
                jx = x + np.random.normal(0, w * 0.06)
                jy = y + np.random.normal(0, h * 0.06)
                jw = w * (1 + np.random.normal(0, 0.07))
                jh = h * (1 + np.random.normal(0, 0.07))
                preds.append({"image_id": img_id, "category_id": cls,
                              "bbox": [round(jx, 2), round(jy, 2), round(jw, 2), round(jh, 2)],
                              "score": round(min(0.99, max(0.05, np.random.beta(6, 2))), 4)})
        # false positives
        for _ in range(np.random.poisson(0.35)):
            preds.append({"image_id": img_id, "category_id": rng.randrange(len(CLASSES)),
                          "bbox": [round(rng.uniform(0, 500), 2), round(rng.uniform(0, 400), 2),
                                   round(rng.uniform(40, 140), 2), round(rng.uniform(40, 140), 2)],
                          "score": round(min(0.9, max(0.05, np.random.beta(2, 5))), 4)})
    return gt, preds


def build():
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(ROOT)

    gt, preds = make_detections()
    write(os.path.join(ROOT, "labels", "val_ground_truth.json"),
          json.dumps({"classes": CLASSES, "annotations": gt}, indent=1))
    write(os.path.join(ROOT, "predictions", "val_predictions.json"),
          json.dumps({"classes": CLASSES, "run_id": CURRENT_RUN, "detections": preds}, indent=1))

    # ---- training config -------------------------------------------------
    write(os.path.join(ROOT, "args.yaml"), f"""\
task: detect
mode: train
model: yolo11s.pt
run_id: {CURRENT_RUN}
data: dataset/dataset.yaml
epochs: 120
imgsz: 640
batch: 16
optimizer: auto
seed: 0
device: 0
""")

    # ---- per-epoch metrics -----------------------------------------------
    rows = []
    for ep in range(1, 121):
        prog = 1 - np.exp(-ep / 28)
        rows.append({
            "epoch": ep,
            "train/box_loss": round(2.10 - 1.35 * prog + np.random.normal(0, 0.012), 5),
            "train/cls_loss": round(3.05 - 2.55 * prog + np.random.normal(0, 0.015), 5),
            "metrics/precision(B)": round(0.32 + 0.53 * prog + np.random.normal(0, 0.006), 5),
            "metrics/recall(B)": round(0.28 + 0.52 * prog + np.random.normal(0, 0.006), 5),
            "metrics/mAP50(B)": round(0.21 + 0.63 * prog + np.random.normal(0, 0.005), 5),
            "metrics/mAP50-95(B)": round(0.11 + 0.47 * prog + np.random.normal(0, 0.004), 5),
        })
    csv_path = os.path.join(ROOT, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- weights (stubs, but hashed like real artifacts) -----------------
    for name in ("best.pt", "last.pt"):
        p = os.path.join(ROOT, "weights", name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(os.urandom(4096))

    # ---- figures: real plots, derived from this package's own data -------
    figdir = os.path.join(ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)

    aps_now = per_class_ap(gt, preds, len(CLASSES))

    plot_confusion_matrix(gt, preds, CLASSES,
                          os.path.join(figdir, "confusion_matrix.png"), CURRENT_RUN)
    plot_pr_curve(gt, preds, CLASSES,
                  os.path.join(figdir, "PR_curve.png"), CURRENT_RUN)

    # DEFECT D1: this figure is plotted from an EARLIER checkpoint's numbers.
    # Its own sidecar records that honestly; the package documents CURRENT_RUN.
    # A separate generator is used so the seeded detection stream is untouched.
    earlier = np.random.default_rng(614).normal(0, 0.055, len(CLASSES))
    aps_earlier = [float(np.clip(a - 0.07 + e, 0.05, 0.99))
                   for a, e in zip(aps_now, earlier)]
    plot_class_performance(aps_earlier, CLASSES,
                           os.path.join(figdir, "fig3_class_performance.png"),
                           EARLIER_RUN)

    for fname, run in (("confusion_matrix.png", CURRENT_RUN),
                       ("PR_curve.png", CURRENT_RUN),
                       ("fig3_class_performance.png", EARLIER_RUN)):
        write(os.path.join(figdir, fname.replace(".png", ".meta.json")),
              json.dumps({"generated_by": "plot_results.py v1.4",
                          "source_run_id": run,
                          "source_file": "results.csv"}, indent=1))

    # ---- dataset descriptor (DEFECT D3: no version / hash / count) -------
    write(os.path.join(ROOT, "dataset", "dataset.yaml"), f"""\
path: ./crime_scene_synth
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
""")

    # ---- manuscript claims (DEFECT D2: overstated mAP) -------------------
    final = rows[-1]
    write(os.path.join(ROOT, "manuscript", "reported_results.json"),
          json.dumps({
              "table_2_caption": "Detection performance on the held-out validation split",
              "reported_mAP50": 0.887,          # overstated
              "reported_precision": round(final["metrics/precision(B)"], 3),
              "reported_recall": round(final["metrics/recall(B)"], 3),
              "figures_cited": ["figures/fig3_class_performance.png",
                                "figures/PR_curve.png"],
          }, indent=1))

    # ---- run manifest: records the weights hashes at training time -------
    manifest = {
        "run_id": CURRENT_RUN,
        "created": "2026-08-02T14:22:10Z",
        "framework": "ultralytics 8.3.x",
        "artifacts": {
            "weights/best.pt": sha256(os.path.join(ROOT, "weights", "best.pt")),
            "weights/last.pt": sha256(os.path.join(ROOT, "weights", "last.pt")),
            "results.csv": sha256(csv_path),
        },
    }
    write(os.path.join(ROOT, "run_manifest.json"), json.dumps(manifest, indent=1))
    print(f"built package at {ROOT}")


if __name__ == "__main__":
    build()
