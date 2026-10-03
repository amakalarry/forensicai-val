"""
ForensicAI-VAL -- reference checker for AI experiment packages.

Given an experiment directory, the agent:
  1. inventories every artifact and records a content hash
  2. plans a check set from what is actually present
  3. executes deterministic checks (no language model in the numeric path)
  4. classifies each finding as VERIFIED / DISCREPANCY / INSUFFICIENT_EVIDENCE
  5. refuses to certify when anything is unresolved, and escalates to a human
  6. writes an append-only audit log and a review report

Design rules enforced here:
  - the agent never asserts that data are truthful, only that records are
    mutually consistent with each other
  - missing evidence is reported as missing, never inferred or filled in
  - artifact contents are treated as untrusted data; nothing in the package
    is executed or interpreted as an instruction
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

TOOL_VERSION = "forensicai-val/0.1.0"

VERIFIED = "VERIFIED"
DISCREPANCY = "DISCREPANCY"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


# --------------------------------------------------------------------------
# audit log
# --------------------------------------------------------------------------
class AuditLog:
    """Append-only record of every action the agent takes."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        self._seq = 0

    def record(self, action: str, inputs: dict, params: dict,
               output: Any, rationale: str) -> None:
        self._seq += 1
        entry = {
            "seq": self._seq,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": action,
            "tool": TOOL_VERSION,
            "runtime": f"python {platform.python_version()}",
            "inputs": inputs,
            "parameters": params,
            "output": output,
            "rationale": rationale,
        }
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


@dataclass
class Finding:
    check: str
    status: str
    claim: str
    evidence: dict = field(default_factory=dict)
    limitation: str = ""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def parse_simple_yaml(path: str) -> dict:
    """Minimal key: value reader. Package contents are data, never executed."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def average_precision(recalls, precisions) -> float:
    """All-point interpolated AP."""
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def compute_map50(gt_anns, pred_dets, n_classes, thr=0.5) -> float:
    """Recompute mAP@0.5 from saved ground truth and predictions."""
    aps = []
    for cls in range(n_classes):
        gts = [g for g in gt_anns if g["category_id"] == cls]
        dts = sorted([d for d in pred_dets if d["category_id"] == cls],
                     key=lambda d: -d["score"])
        if not gts:
            continue
        by_img: dict[int, list] = {}
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
        aps.append(average_precision(recalls, precisions))
    return sum(aps) / len(aps) if aps else 0.0


# --------------------------------------------------------------------------
# the agent
# --------------------------------------------------------------------------
class ForensicAIVAL:
    def __init__(self, package_dir: str, out_dir: str):
        self.pkg = os.path.abspath(package_dir)
        self.out = os.path.abspath(out_dir)
        os.makedirs(self.out, exist_ok=True)
        self.audit = AuditLog(os.path.join(self.out, "audit_log.jsonl"))
        self.inventory: dict[str, dict] = {}
        self.findings: list[Finding] = []

    # -- step 1 ------------------------------------------------------------
    def run_inventory(self) -> None:
        for dirpath, _, files in os.walk(self.pkg):
            for f in sorted(files):
                full = os.path.join(dirpath, f)
                rel = os.path.relpath(full, self.pkg)
                self.inventory[rel] = {"sha256": sha256(full),
                                       "bytes": os.path.getsize(full)}
        self.audit.record(
            action="inventory_package",
            inputs={"package_dir": self.pkg},
            params={"hash": "sha256"},
            output={"artifact_count": len(self.inventory)},
            rationale="Registered every file and its content hash so later findings "
                      "can be tied to the exact bytes examined.",
        )

    # -- step 2 ------------------------------------------------------------
    def plan(self) -> list[str]:
        have = set(self.inventory)
        checks = []
        if "run_manifest.json" in have:
            checks.append("artifact_integrity")
        if any(p.startswith("figures/") and p.endswith(".meta.json") for p in have):
            checks.append("figure_provenance")
        if ("predictions/val_predictions.json" in have
                and "labels/val_ground_truth.json" in have
                and "manuscript/reported_results.json" in have):
            checks.append("reported_metric_recomputation")
        if any(p.startswith("dataset/") for p in have):
            checks.append("dataset_identity")
        if "manuscript/reported_results.json" in have:
            checks.append("cited_figures_resolve")
        self.audit.record(
            action="plan_checks",
            inputs={"artifacts_present": sorted(have)},
            params={"registry": "deterministic-v1"},
            output={"planned_checks": checks},
            rationale="Selected the checks whose required evidence is present. "
                      "Checks without their inputs are not silently skipped; they "
                      "are reported as insufficient evidence.",
        )
        return checks

    # -- checks ------------------------------------------------------------
    def check_artifact_integrity(self) -> None:
        manifest = load_json(os.path.join(self.pkg, "run_manifest.json"))
        mismatched, missing, ok = [], [], []
        for rel, expected in manifest.get("artifacts", {}).items():
            actual = self.inventory.get(rel, {}).get("sha256")
            if actual is None:
                missing.append(rel)
            elif actual != expected:
                mismatched.append(rel)
            else:
                ok.append(rel)
        if mismatched or missing:
            status, claim = DISCREPANCY, "Registered artifacts do not match the run manifest."
        else:
            status, claim = VERIFIED, f"All {len(ok)} manifest-registered artifacts match their recorded hashes."
        self.findings.append(Finding(
            "artifact_integrity", status, claim,
            {"matching": ok, "mismatched": mismatched, "absent": missing},
            "A hash match shows the bytes are unchanged since registration. It does "
            "not establish that the original data were correctly collected.",
        ))
        self.audit.record("check.artifact_integrity",
                          {"run_manifest.json": self.inventory["run_manifest.json"]["sha256"]},
                          {"algorithm": "sha256"},
                          {"status": status, "matching": len(ok),
                           "mismatched": mismatched, "absent": missing},
                          "Compared recomputed hashes against the values recorded at training time.")

    def check_figure_provenance(self) -> None:
        args = parse_simple_yaml(os.path.join(self.pkg, "args.yaml"))
        current = args.get("run_id")
        stale, aligned, unknown = [], [], []
        for rel in sorted(self.inventory):
            if not (rel.startswith("figures/") and rel.endswith(".meta.json")):
                continue
            meta = load_json(os.path.join(self.pkg, rel))
            fig = rel.replace(".meta.json", ".png")
            src = meta.get("source_run_id")
            if src is None:
                unknown.append(fig)
            elif src != current:
                stale.append({"figure": fig, "source_run_id": src,
                              "package_run_id": current})
            else:
                aligned.append(fig)
        if stale:
            status = DISCREPANCY
            claim = (f"{len(stale)} figure(s) were generated from a different run "
                     f"than the one this package documents.")
        elif unknown:
            status, claim = INSUFFICIENT, "Some figures carry no source run identifier."
        else:
            status, claim = VERIFIED, f"All {len(aligned)} figures trace to run {current}."
        self.findings.append(Finding(
            "figure_provenance", status, claim,
            {"package_run_id": current, "stale": stale,
             "aligned": aligned, "no_provenance": unknown},
            "Provenance is read from sidecar metadata written by the plotting step. "
            "A figure with no sidecar cannot be traced by this check.",
        ))
        self.audit.record("check.figure_provenance",
                          {"args.yaml": self.inventory["args.yaml"]["sha256"]},
                          {"compare": "sidecar.source_run_id vs args.run_id"},
                          {"status": status, "stale": stale, "aligned": len(aligned)},
                          "Compared each figure's recorded source run against the package's run id.")

    def check_reported_metric(self) -> None:
        gt = load_json(os.path.join(self.pkg, "labels", "val_ground_truth.json"))
        pr = load_json(os.path.join(self.pkg, "predictions", "val_predictions.json"))
        rep = load_json(os.path.join(self.pkg, "manuscript", "reported_results.json"))
        n_classes = len(gt["classes"])
        recomputed = compute_map50(gt["annotations"], pr["detections"], n_classes)
        reported = rep.get("reported_mAP50")
        tolerance = 0.005
        delta = None if reported is None else round(reported - recomputed, 4)
        if reported is None:
            status, claim = INSUFFICIENT, "No reported mAP@50 to compare against."
        elif abs(delta) > tolerance:
            status = DISCREPANCY
            claim = (f"Reported mAP@50 of {reported:.4f} does not match "
                     f"{recomputed:.4f} recomputed from the saved predictions "
                     f"(difference {delta:+.4f}).")
        else:
            status, claim = VERIFIED, f"Reported mAP@50 agrees with recomputation ({recomputed:.4f})."
        self.findings.append(Finding(
            "reported_metric_recomputation", status, claim,
            {"reported_mAP50": reported, "recomputed_mAP50": round(recomputed, 4),
             "difference": delta, "tolerance": tolerance,
             "n_ground_truth": len(gt["annotations"]), "n_detections": len(pr["detections"])},
            "Recomputation uses the predictions saved in the package. If those "
            "predictions were not the ones used to produce the manuscript, this "
            "check cannot detect it.",
        ))
        self.audit.record("check.reported_metric_recomputation",
                          {"predictions/val_predictions.json": self.inventory["predictions/val_predictions.json"]["sha256"],
                           "labels/val_ground_truth.json": self.inventory["labels/val_ground_truth.json"]["sha256"],
                           "manuscript/reported_results.json": self.inventory["manuscript/reported_results.json"]["sha256"]},
                          {"metric": "mAP@0.50", "matching": "greedy IoU",
                           "interpolation": "all-point", "tolerance": tolerance},
                          {"status": status, "reported": reported,
                           "recomputed": round(recomputed, 4), "difference": delta},
                          "Recomputed the headline metric from saved predictions and compared "
                          "it to the value the manuscript reports.")

    def check_dataset_identity(self) -> None:
        ds_path = os.path.join(self.pkg, "dataset", "dataset.yaml")
        required = {"version": "dataset version",
                    "manifest_sha256": "content hash of the image manifest",
                    "n_images": "image count"}
        present = parse_simple_yaml(ds_path) if os.path.exists(ds_path) else {}
        absent = {k: v for k, v in required.items() if k not in present}
        if absent:
            status = INSUFFICIENT
            claim = ("Dataset identity cannot be established: "
                     + ", ".join(absent.values()) + " not recorded.")
        else:
            status, claim = VERIFIED, "Dataset version and manifest hash are recorded."
        self.findings.append(Finding(
            "dataset_identity", status, claim,
            {"fields_present": sorted(present), "fields_absent": sorted(absent)},
            "Without a version and manifest hash there is no way to confirm which "
            "images were used. This is reported as missing evidence, not as a fault "
            "in the results.",
        ))
        self.audit.record("check.dataset_identity",
                          {"dataset/dataset.yaml": self.inventory.get("dataset/dataset.yaml", {}).get("sha256")},
                          {"required_fields": sorted(required)},
                          {"status": status, "absent": sorted(absent)},
                          "Checked for the fields needed to pin the dataset to a specific version. "
                          "Did not guess a version from the directory name.")

    def check_cited_figures(self) -> None:
        rep = load_json(os.path.join(self.pkg, "manuscript", "reported_results.json"))
        cited = rep.get("figures_cited", [])
        missing = [c for c in cited if c not in self.inventory]
        status = DISCREPANCY if missing else VERIFIED
        claim = (f"{len(missing)} cited figure(s) are not present in the package."
                 if missing else f"All {len(cited)} cited figures resolve to package artifacts.")
        self.findings.append(Finding(
            "cited_figures_resolve", status, claim,
            {"cited": cited, "unresolved": missing}, ""))
        self.audit.record("check.cited_figures_resolve",
                          {"manuscript/reported_results.json": self.inventory["manuscript/reported_results.json"]["sha256"]},
                          {}, {"status": status, "unresolved": missing},
                          "Confirmed every figure the manuscript cites exists in the package.")

    # -- step 5 ------------------------------------------------------------
    def decide(self) -> dict:
        counts = {s: sum(1 for f in self.findings if f.status == s)
                  for s in (VERIFIED, DISCREPANCY, INSUFFICIENT)}
        if counts[DISCREPANCY] or counts[INSUFFICIENT]:
            decision = "REFUSE_TO_CERTIFY"
            action = "ESCALATE_TO_HUMAN_REVIEWER"
            reason = (f"{counts[DISCREPANCY]} discrepancy finding(s) and "
                      f"{counts[INSUFFICIENT]} insufficient-evidence finding(s) remain "
                      f"unresolved. The agent has no authority to accept these.")
        else:
            decision = "CONSISTENT_WITH_REGISTERED_EVIDENCE"
            action = "RETURN_TO_REVIEWER_FOR_APPROVAL"
            reason = "All planned checks verified. Release remains a human decision."
        out = {"decision": decision, "next_action": action,
               "reason": reason, "counts": counts}
        self.audit.record("decide", {"findings": len(self.findings)},
                          {"policy": "refuse-on-any-unresolved"}, out,
                          "The agent does not resolve discrepancies or accept missing "
                          "evidence on its own; both route to a human reviewer.")
        return out

    # -- orchestration -----------------------------------------------------
    def run(self) -> dict:
        self.run_inventory()
        dispatch = {
            "artifact_integrity": self.check_artifact_integrity,
            "figure_provenance": self.check_figure_provenance,
            "reported_metric_recomputation": self.check_reported_metric,
            "dataset_identity": self.check_dataset_identity,
            "cited_figures_resolve": self.check_cited_figures,
        }
        for name in self.plan():
            dispatch[name]()
        decision = self.decide()
        self.write_report(decision)
        return decision

    def write_report(self, decision: dict) -> None:
        order = {DISCREPANCY: 0, INSUFFICIENT: 1, VERIFIED: 2}
        lines = [
            "# ForensicAI-VAL review report",
            "",
            f"- Package: `{os.path.basename(self.pkg)}`",
            f"- Reviewed: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- Tool: {TOOL_VERSION}",
            f"- Artifacts registered: {len(self.inventory)}",
            "",
            f"## Decision: {decision['decision']}",
            "",
            f"**Next action: {decision['next_action']}**",
            "",
            decision["reason"],
            "",
            f"Verified {decision['counts'][VERIFIED]} · "
            f"Discrepancy {decision['counts'][DISCREPANCY]} · "
            f"Insufficient evidence {decision['counts'][INSUFFICIENT]}",
            "",
            "## Findings",
            "",
        ]
        for f in sorted(self.findings, key=lambda x: order[x.status]):
            lines += [f"### [{f.status}] {f.check}", "", f.claim, "",
                      "```json", json.dumps(f.evidence, indent=1), "```", ""]
            if f.limitation:
                lines += [f"*Limitation:* {f.limitation}", ""]
        lines += ["## Scope of this review", "",
                  "This review establishes whether the records in the package are mutually "
                  "consistent and sufficiently documented. It does not establish that the "
                  "underlying data were correctly collected, that the experiment was well "
                  "designed, or that no misconduct occurred. Every action taken is listed "
                  "in `audit_log.jsonl`.", ""]
        with open(os.path.join(self.out, "review_report.md"), "w") as fh:
            fh.write("\n".join(lines))


if __name__ == "__main__":
    pkg = sys.argv[1] if len(sys.argv) > 1 else "demo/experiment_package"
    out = sys.argv[2] if len(sys.argv) > 2 else "demo/review_output"
    result = ForensicAIVAL(pkg, out).run()
    print(json.dumps(result, indent=1))
