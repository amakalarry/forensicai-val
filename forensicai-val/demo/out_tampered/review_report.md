# ForensicAI-VAL review report

- Package: `package_tampered`
- Reviewed: 2026-10-03T22:48:21+00:00
- Tool: forensicai-val/0.1.0
- Artifacts registered: 15

## Decision: REFUSE_TO_CERTIFY

**Next action: ESCALATE_TO_HUMAN_REVIEWER**

3 discrepancy finding(s) and 1 insufficient-evidence finding(s) remain unresolved. The agent has no authority to accept these.

Verified 1 · Discrepancy 3 · Insufficient evidence 1

## Findings

### [DISCREPANCY] artifact_integrity

Registered artifacts do not match the run manifest.

```json
{
 "matching": [
  "weights/last.pt",
  "results.csv"
 ],
 "mismatched": [
  "weights/best.pt"
 ],
 "absent": []
}
```

*Limitation:* A hash match shows the bytes are unchanged since registration. It does not establish that the original data were correctly collected.

### [DISCREPANCY] figure_provenance

1 figure(s) were generated from a different run than the one this package documents.

```json
{
 "package_run_id": "run_2026-08-02_9c7d",
 "stale": [
  {
   "figure": "figures/fig3_class_performance.png",
   "source_run_id": "run_2026-06-14_a3f1",
   "package_run_id": "run_2026-08-02_9c7d"
  }
 ],
 "aligned": [
  "figures/PR_curve.png",
  "figures/confusion_matrix.png"
 ],
 "no_provenance": []
}
```

*Limitation:* Provenance is read from sidecar metadata written by the plotting step. A figure with no sidecar cannot be traced by this check.

### [DISCREPANCY] reported_metric_recomputation

Reported mAP@50 of 0.8870 does not match 0.8569 recomputed from the saved predictions (difference +0.0301).

```json
{
 "reported_mAP50": 0.887,
 "recomputed_mAP50": 0.8569,
 "difference": 0.0301,
 "tolerance": 0.005,
 "n_ground_truth": 572,
 "n_detections": 569
}
```

*Limitation:* Recomputation uses the predictions saved in the package. If those predictions were not the ones used to produce the manuscript, this check cannot detect it.

### [INSUFFICIENT_EVIDENCE] dataset_identity

Dataset identity cannot be established: dataset version, content hash of the image manifest, image count not recorded.

```json
{
 "fields_present": [
  "names",
  "nc",
  "path",
  "train",
  "val"
 ],
 "fields_absent": [
  "manifest_sha256",
  "n_images",
  "version"
 ]
}
```

*Limitation:* Without a version and manifest hash there is no way to confirm which images were used. This is reported as missing evidence, not as a fault in the results.

### [VERIFIED] cited_figures_resolve

All 2 cited figures resolve to package artifacts.

```json
{
 "cited": [
  "figures/fig3_class_performance.png",
  "figures/PR_curve.png"
 ],
 "unresolved": []
}
```

## Scope of this review

This review establishes whether the records in the package are mutually consistent and sufficiently documented. It does not establish that the underlying data were correctly collected, that the experiment was well designed, or that no misconduct occurred. Every action taken is listed in `audit_log.jsonl`.
