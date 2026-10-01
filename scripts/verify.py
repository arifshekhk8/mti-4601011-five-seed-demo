"""Gate all video inference against the frozen controlled comparison record."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from common import CONFIG, ROOT, SEEDS, THRESHOLD, load_ensemble, load_splits, predict_per_seed, sha256


def metrics(y, p, tau):
    return {
        "f1_tau": float(f1_score(y, p >= tau)),
        "f1_05": float(f1_score(y, p >= 0.5)),
        "acc": float(accuracy_score(y, p >= tau)),
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": recorded_pr_auc(y, p),
    }


def recorded_pr_auc(y, p):
    """The source comparison's stable descending-rank AP, including its tie rule."""
    yy = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(p), kind="mergesort")
    ranked = yy[order]
    tp = np.cumsum(ranked)
    fp = np.cumsum(~ranked)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(yy.sum()), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def best_threshold(y, p):
    candidates = np.unique(p)
    candidates = candidates[(candidates >= 0.05) & (candidates <= 0.95)]
    return float(max(candidates, key=lambda t: (f1_score(y, p >= t), -abs(t - 0.5), t)))


def main():
    reference = json.loads((ROOT / "private/source/matched_comparison_results.json").read_text())
    expected = reference["families"]["BiLSTM"]
    assert reference["protocol"]["seeds"] == SEEDS
    assert expected["cfg"] == {"lr": 0.0001, "dropout": 0.2, "hidden": 256, "num_layers": 2}
    assert expected["n_params"] == 2237313
    assert expected["tau"] == THRESHOLD
    hashes = {}
    for seed in SEEDS:
        d = ROOT / "private/checkpoints" / f"seed{seed}"
        final = json.loads((d / "final.json").read_text())
        assert final["seed"] == seed and final["family"] == "bilstm"
        assert final["cfg"] == expected["cfg"] and final["select"] == "auc"
        assert final["pos_weight"] == 1.682 and final["device"] == "cpu"
        assert final["n_params"] == 2237313
        hashes[f"seed{seed}"] = {
            f.name: sha256(f) for f in sorted(d.iterdir()) if f.name in ("best.pt", "final.json", "norm_mean.npy", "norm_std.npy")
        }
    xtr, ytr, xva, yva, xte, yte, _ = load_splits()
    models, stats = load_ensemble("cpu")
    for mean, std in stats:
        np.testing.assert_allclose(mean, xtr.mean((0, 1)), rtol=0, atol=2e-5)
        np.testing.assert_allclose(std, xtr.std((0, 1)) + 1e-6, rtol=0, atol=2e-5)
    pva = predict_per_seed(models, stats, xva)
    pte = predict_per_seed(models, stats, xte)
    tau = best_threshold(yva, pva.mean(axis=0))
    assert abs(tau - THRESHOLD) < 1e-6, (tau, THRESHOLD)
    per = [metrics(yte, p, THRESHOLD) for p in pte]
    ensemble = metrics(yte, pte.mean(axis=0), THRESHOLD)
    tolerance = 1e-5
    diffs = {}
    for i, seed in enumerate(SEEDS):
        diffs[f"seed{seed}"] = {k: abs(per[i][k] - expected["per_seed"][i][k]) for k in per[i]}
    mean_sd = {k: [float(np.mean([m[k] for m in per])), float(np.std([m[k] for m in per], ddof=1))] for k in per[0]}
    diffs["per_seed_mean_sd"] = {k: [abs(v[0]-expected["mean_sd"][k][0]), abs(v[1]-expected["mean_sd"][k][1])] for k,v in mean_sd.items()}
    diffs["ensemble"] = {k: abs(ensemble[k] - expected["ens"][k]) for k in expected["ens"]}
    maximum = max([v for row in diffs.values() for value in row.values() for v in (value if isinstance(value, list) else [value])])
    passed = maximum < tolerance
    report = {
        "pass": passed, "tolerance": tolerance, "maximum_absolute_metric_difference": maximum,
        "data_counts": {"train": len(ytr), "validation": len(yva), "test": len(yte)},
        "threshold_recorded": THRESHOLD, "threshold_recomputed_validation": tau,
        "per_seed": {f"seed{s}": p for s, p in zip(SEEDS, per)},
        "per_seed_mean_sd": mean_sd, "ensemble": ensemble,
        "reference_per_seed_mean_sd": expected["mean_sd"], "reference_ensemble": expected["ens"],
        "absolute_differences": diffs, "checkpoint_sha256": hashes,
        "data_sha256": {f.name: sha256(f) for f in (ROOT / "private/data").iterdir() if f.is_file()},
        "record_sha256": sha256(ROOT / "private/source/matched_comparison_results.json"),
    }
    (ROOT / "results/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "results/verification_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["type", "seed", "f1_tau", "f1_05", "acc", "auc", "pr_auc"], lineterminator="\n")
        w.writeheader()
        for seed, m in zip(SEEDS, per): w.writerow({"type": "individual", "seed": seed, **m})
        w.writerow({"type": "ensemble", "seed": "mean probabilities", **ensemble})
    lines = ["# Quantitative parity gate", "", f"**Status: {'PASS' if passed else 'FAIL'}**. Maximum absolute metric difference {maximum:.3g}; tolerance {tolerance:g}.", "",
             "The test set contains 2,094 windows. The pooled validation threshold was recomputed from the five seed probability vectors and equals the recorded 0.4579877257347107. No threshold or checkpoint was reselected on test or video.", "",
             "| Quantity | Recomputed | Recorded |", "|---|---:|---:|",
             f"| Ensemble F1 at tau | {ensemble['f1_tau']:.6f} | {expected['ens']['f1_tau']:.6f} |",
             f"| Ensemble ROC AUC | {ensemble['auc']:.6f} | {expected['ens']['auc']:.6f} |",
             f"| Ensemble PR AUC | {ensemble['pr_auc']:.6f} | {expected['ens']['pr_auc']:.6f} |",
             f"| Per-seed mean F1 at tau | {mean_sd['f1_tau'][0]:.6f} | {expected['mean_sd']['f1_tau'][0]:.6f} |",
             f"| Per-seed mean ROC AUC | {mean_sd['auc'][0]:.6f} | {expected['mean_sd']['auc'][0]:.6f} |",
             f"| Per-seed mean PR AUC | {mean_sd['pr_auc'][0]:.6f} | {expected['mean_sd']['pr_auc'][0]:.6f} |", "",
             "Ensemble metrics are calculated from the mean probability vector. Per-seed means are the arithmetic mean of five independently scored checkpoints. Their values are not interchangeable.", "",
             "Checkpoint, normalization, source record, and data SHA-256 digests are in `results/verification.json`. The private input files are excluded from the public repository."]
    (ROOT / "reports/verification.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"pass": passed, "tau": tau, "ensemble": ensemble, "mean_sd": mean_sd, "max_diff": maximum}, indent=2))
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    main()
