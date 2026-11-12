import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless research environments
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, Any, List, Optional


def generate_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str = "outputs",
    prefix: str = "test",
    y_prob_calibrated: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, str]:
    """
    Generates research-grade evaluation figures:
    1. ROC Curve
    2. Precision-Recall Curve
    3. Calibration / Reliability Diagram (Pre vs. Post Temperature Scaling)
    4. Confusion Matrix Heatmap
    Returns: Dict mapping plot name to absolute file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"Proposed Model (AUC = {np.trapz(tpr, fpr):.3f})")
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", lw=1.5, label="Random Guess")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)
    generated_files["roc_curve"] = roc_path

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(rec, prec, color="#059669", lw=2.5, label=f"Proposed Model (AUPRC = {np.trapz(prec[::-1], rec[::-1]):.3f})")
    prevalence = np.mean(y_true)
    ax.axhline(prevalence, color="#94A3B8", linestyle="--", lw=1.5, label=f"Baseline Prevalence ({prevalence:.1%})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    pr_path = os.path.join(output_dir, f"{prefix}_pr_curve.png")
    fig.tight_layout()
    fig.savefig(pr_path)
    plt.close(fig)
    generated_files["pr_curve"] = pr_path

    # 3. Calibration / Reliability Diagram
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", label="Perfect Calibration")

    def calc_bins(probs):
        bins = np.linspace(0.0, 1.0, 11)
        bin_confs, bin_accs = [], []
        for b in range(10):
            m = (probs >= bins[b]) & (probs < bins[b + 1])
            if np.any(m):
                bin_confs.append(np.mean(probs[m]))
                bin_accs.append(np.mean(y_true[m]))
        return bin_confs, bin_accs

    confs_raw, accs_raw = calc_bins(y_prob)
    ax.plot(confs_raw, accs_raw, marker="o", lw=2, color="#DC2626", label="Uncalibrated")

    if y_prob_calibrated is not None:
        confs_cal, accs_cal = calc_bins(y_prob_calibrated)
        ax.plot(confs_cal, accs_cal, marker="s", lw=2, color="#2563EB", label="Temperature Scaled")

    ax.set_xlabel("Mean Predicted Confidence", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed Clinical Fraction", fontsize=11, fontweight="bold")
    ax.set_title("Reliability Diagram (Calibration)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    cal_path = os.path.join(output_dir, f"{prefix}_calibration_plot.png")
    fig.tight_layout()
    fig.savefig(cal_path)
    plt.close(fig)
    generated_files["calibration_plot"] = cal_path

    # 4. Confusion Matrix Heatmap
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Negative (No Event)", "Positive (HF Event)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Confusion Matrix (Threshold={threshold:.2f})",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold", fontsize=12
            )
    cm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)
    generated_files["confusion_matrix"] = cm_path

    return generated_files
