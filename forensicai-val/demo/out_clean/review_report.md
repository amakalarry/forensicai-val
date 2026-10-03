# ForensicAI-VAL review report

- Package: `package_clean`
- Reviewed: 2026-10-03T22:48:21+00:00
- Tool: forensicai-val/0.1.0
- Artifacts registered: 15

## Decision: CONSISTENT_WITH_REGISTERED_EVIDENCE

**Next action: RETURN_TO_REVIEWER_FOR_APPROVAL**

All planned checks verified. Release remains a human decision.

Verified 5 · Discrepancy 0 · Insufficient evidence 0

## Findings

### [VERIFIED] artifact_integrity

All 3 manifest-registered artifacts match their recorded hashes.

```json
{
 "matching": [
  "weights/best.pt",
  "weights/last.pt",
  "results.csv"
 ],
 "mismatched": [],
 "absent": []
}
```

*Limitation:* A hash match shows the bytes are unchanged since registration. It does not establish that the original data were correctly collected.

### [VERIFIED] figure_provenance

All 3 figures trace to run run_2026-08-02_9c7d.

```json
{
 "package_run_id": "run_2026-08-02_9c7d",
 "stale": [],
 "aligned": [
  "figures/PR_curve.png",
  "figures/confusion_matrix.png",
  "figures/fig3_class_performance.png"
 ],
 "no_provenance": []
}
```

*Limitation:* Provenance is read from sidecar metadata written by the plotting step. A figure with no sidecar cannot be traced by this check.

### [VERIFIED] reported_metric_recomputation

Reported mAP@50 agrees with recomputation (0.8569).

```json
{
 "reported_mAP50": 0.8569,
 "recomputed_mAP50": 0.8569,
 "difference": 0.0,
 "tolerance": 0.005,
 "n_ground_truth": 572,
 "n_detections": 569
}
```

*Limitation:* Recomputation uses the predictions saved in the package. If those predictions were not the ones used to produce the manuscript, this check cannot detect it.

### [VERIFIED] dataset_identity

Dataset version and manifest hash are recorded.

```json
{
 "fields_present": [
  "manifest_sha256",
  "n_images",
  "names",
  "nc",
  "path",
  "train",
  "val",
  "version"
 ],
 "fields_absent": []
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
