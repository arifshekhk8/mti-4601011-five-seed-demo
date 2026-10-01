# Quantitative parity gate

**Status: PASS**. Maximum absolute metric difference 0; tolerance 1e-05.

The test set contains 2,094 windows. The pooled validation threshold was recomputed from the five seed probability vectors and equals the recorded 0.4579877257347107. No threshold or checkpoint was reselected on test or video.

| Quantity | Recomputed | Recorded |
|---|---:|---:|
| Ensemble F1 at tau | 0.831334 | 0.831334 |
| Ensemble ROC AUC | 0.931970 | 0.931970 |
| Ensemble PR AUC | 0.876826 | 0.876826 |
| Per-seed mean F1 at tau | 0.827587 | 0.827587 |
| Per-seed mean ROC AUC | 0.924225 | 0.924225 |
| Per-seed mean PR AUC | 0.868774 | 0.868774 |

Ensemble metrics are calculated from the mean probability vector. Per-seed means are the arithmetic mean of five independently scored checkpoints. Their values are not interchangeable.

Checkpoint, normalization, source record, and data SHA-256 digests are in `results/verification.json`. The private input files are excluded from the public repository.
