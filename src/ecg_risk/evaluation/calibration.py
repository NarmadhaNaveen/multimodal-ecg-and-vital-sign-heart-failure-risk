import numpy as np
import scipy.optimize as optimize
from typing import Tuple, Dict, Any
from .metrics import compute_ece, compute_brier_score


class TemperatureScaler:
    """
    Post-hoc probability calibration via Temperature Scaling (Guo et al., ICML 2017).
    Calibrates model confidence by optimizing a single scalar parameter T > 0
    on validation set negative log likelihood.
    
    T_scaled_logits = logits / T
    calibrated_probs = sigmoid(logits / T)
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Fits temperature parameter on validation cohort only.
        val_logits: 1D array of unnormalized logits
        val_labels: 1D array of binary true outcomes (0 or 1)
        """
        val_logits = np.asarray(val_logits, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.float64)

        def nll_loss(t_param: np.ndarray) -> float:
            T = t_param[0]
            if T <= 0.01:
                return 1e9
            scaled_logits = val_logits / T
            # Log-loss: - y*log(p) - (1-y)*log(1-p)
            # Numerically stable: log(1 + exp(-scaled_logits)) etc.
            loss = np.maximum(scaled_logits, 0) - scaled_logits * val_labels + np.log1p(np.exp(-np.abs(scaled_logits)))
            return float(np.mean(loss))

        res = optimize.minimize(
            nll_loss,
            x0=[1.0],
            bounds=[(0.05, 10.0)],
            method="L-BFGS-B",
        )

        self.temperature = float(res.x[0])
        self.is_fitted = True
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Calibrates logits and returns calibrated probabilities."""
        scaled_logits = np.asarray(logits, dtype=np.float64) / self.temperature
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        return probs.astype(np.float32)

    def evaluate_calibration_impact(
        self,
        test_logits: np.ndarray,
        test_labels: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Compares pre-calibration and post-calibration ECE and Brier scores on test set.
        """
        raw_probs = 1.0 / (1.0 + np.exp(-np.asarray(test_logits, dtype=np.float64)))
        cal_probs = self.calibrate(test_logits)

        ece_before = compute_ece(test_labels, raw_probs, n_bins=n_bins)
        ece_after = compute_ece(test_labels, cal_probs, n_bins=n_bins)

        brier_before = compute_brier_score(test_labels, raw_probs)
        brier_after = compute_brier_score(test_labels, cal_probs)

        return {
            "temperature": self.temperature,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "ece_reduction_percent": ((ece_before - ece_after) / (ece_before + 1e-8)) * 100.0,
            "brier_reduction_percent": ((brier_before - brier_after) / (brier_before + 1e-8)) * 100.0,
        }
