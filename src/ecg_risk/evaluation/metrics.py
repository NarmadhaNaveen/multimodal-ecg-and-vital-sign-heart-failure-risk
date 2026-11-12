import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from typing import Dict, Any, Tuple, Optional


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Divides predicted probabilities into n_bins equal-width bins,
    calculates the absolute difference between average accuracy and average confidence.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n_samples = len(y_true)

    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculates Mean Squared Error of probability forecasts (Brier score)."""
    return float(brier_score_loss(y_true, y_prob))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
) -> float:
    """
    Determines optimal classification threshold strictly using validation data.
    Methods:
    - 'youden': Maximizes Sensitivity + Specificity - 1 (Youden's J)
    - 'f1': Maximizes F1 score on validation cohort
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_score = -1.0
    best_thresh = 0.5

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        if method == "youden":
            sens = recall_score(y_true, preds, zero_division=0)
            tn = np.sum((y_true == 0) & (preds == 0))
            fp = np.sum((y_true == 0) & (preds == 1))
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = sens + spec - 1.0
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = float(th)

    return best_thresh


def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: Optional[float] = None,
    logits: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes comprehensive clinical metrics on predictions.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_probs = np.asarray(y_probs, dtype=np.float32)

    # Threshold determination
    if threshold is None:
        threshold = 0.5

    y_pred = (y_probs >= threshold).astype(np.int32)

    # AUROC
    try:
        auroc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auroc = 0.5

    # AUPRC
    try:
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_probs)
        auprc = float(auc(recall_arr, precision_arr)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        auprc = 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    ece = compute_ece(y_true, y_probs, n_bins=n_bins)
    brier = compute_brier_score(y_true, y_probs)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "sensitivity": rec,
        "specificity": spec,
        "f1_score": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "ece": ece,
        "brier_score": brier,
        "threshold": float(threshold),
        "num_samples": len(y_true),
        "prevalence": float(np.mean(y_true)),
    }
