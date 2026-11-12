import numpy as np
from typing import Dict, Any, List, Optional
from .metrics import evaluate_predictions


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: Optional[List[str]] = None,
    n_bootstraps: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Computes patient-level cluster bootstrap confidence intervals.
    If patient_ids are provided, resamples clusters of patient observations together,
    accurately reflecting clinical intra-patient correlation.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_samples = len(y_true)

    boot_metrics: Dict[str, List[float]] = {
        "auroc": [],
        "auprc": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "f1_score": [],
        "ece": [],
        "brier_score": [],
    }

    if patient_ids is not None:
        unique_pids = np.unique(patient_ids)
        pid_to_indices = {pid: np.where(np.array(patient_ids) == pid)[0] for pid in unique_pids}
        n_clusters = len(unique_pids)

        for _ in range(n_bootstraps):
            resampled_pids = rng.choice(unique_pids, size=n_clusters, replace=True)
            resampled_idx = np.concatenate([pid_to_indices[p] for p in resampled_pids])
            sub_y = y_true[resampled_idx]
            sub_p = y_prob[resampled_idx]

            if len(np.unique(sub_y)) < 2:
                continue

            res = evaluate_predictions(sub_y, sub_p, threshold=threshold)
            for k in boot_metrics:
                if k in res:
                    boot_metrics[k].append(res[k])
    else:
        for _ in range(n_bootstraps):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            sub_y = y_true[idx]
            sub_p = y_prob[idx]

            if len(np.unique(sub_y)) < 2:
                continue

            res = evaluate_predictions(sub_y, sub_p, threshold=threshold)
            for k in boot_metrics:
                if k in res:
                    boot_metrics[k].append(res[k])

    results_ci = {}
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)

    for k, values in boot_metrics.items():
        if len(values) > 0:
            arr = np.array(values)
            results_ci[k] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "ci_lower": float(np.percentile(arr, lower_pct)),
                "ci_upper": float(np.percentile(arr, upper_pct)),
            }

    return results_ci
