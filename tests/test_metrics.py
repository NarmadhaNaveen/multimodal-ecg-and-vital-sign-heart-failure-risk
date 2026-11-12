import pytest
import numpy as np
from src.ecg_risk.evaluation.metrics import (
    evaluate_predictions,
    compute_ece,
    compute_brier_score,
    find_optimal_threshold,
)
from src.ecg_risk.evaluation.calibration import TemperatureScaler
from src.ecg_risk.evaluation.confidence_intervals import bootstrap_metric_ci


def test_metrics_evaluation_outputs():
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.85])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)

    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "accuracy" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "f1_score" in metrics
    assert "ece" in metrics
    assert "brier_score" in metrics
    assert "confusion_matrix" in metrics

    assert metrics["auroc"] >= 0.9
    assert 0.0 <= metrics["ece"] <= 1.0


def test_optimal_threshold_selection_on_validation():
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    y_prob = np.array([0.65, 0.7, 0.2, 0.3, 0.8, 0.1, 0.25, 0.35])

    thresh = find_optimal_threshold(y_true, y_prob, method="youden")
    assert 0.3 <= thresh <= 0.65


def test_temperature_scaling_calibration():
    # Simulated overconfident logits
    logits = np.array([4.0, -4.0, 3.5, -3.0, 3.8, -2.5, -3.2, 3.1])
    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    scaler = TemperatureScaler()
    t = scaler.fit(logits, labels)
    assert t > 0.0
    assert scaler.is_fitted

    cal_probs = scaler.calibrate(logits)
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))

    impact = scaler.evaluate_calibration_impact(logits, labels)
    assert "ece_before" in impact
    assert "ece_after" in impact


def test_bootstrap_confidence_intervals():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
    y_prob = np.array([0.85, 0.15, 0.8, 0.2, 0.75, 0.25, 0.9, 0.1, 0.7, 0.3] * 5)
    pids = [f"p_{i//2}" for i in range(len(y_true))]

    ci = bootstrap_metric_ci(y_true, y_prob, patient_ids=pids, n_bootstraps=50)
    assert "auroc" in ci
    assert "ci_lower" in ci["auroc"]
    assert "ci_upper" in ci["auroc"]
    assert ci["auroc"]["ci_lower"] <= ci["auroc"]["ci_upper"]
