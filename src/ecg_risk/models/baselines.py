import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class LogisticRegressionEngineeredBaseline:
    """
    Reproducible Clinical Baseline:
    Logistic Regression fitted on clinically defensible engineered summary features
    extracted from the ECG waveforms and vital-sign time series.
    
    Features:
    - Vitals: mean, std, min, max, delta (t_end - t_start) for each channel (HR, BP, SpO2, Resp)
    - Missingness: fraction of missing samples per channel
    - ECG: root mean square (RMS) energy, mean amplitude, peak-to-peak amplitude per lead
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced")
        self.is_fitted = False

    def extract_features(self, ecg: np.ndarray, vitals: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        ecg: (C_ecg, T_ecg)
        vitals: (C_vital, T_vital)
        mask: (C_vital, T_vital)
        Returns: 1D feature vector
        """
        feats = []

        # Vital sign summary features
        C_v, T_v = vitals.shape
        for c in range(C_v):
            v_ch = vitals[c]
            m_ch = mask[c]
            obs_vals = v_ch[m_ch > 0.5] if np.any(m_ch > 0.5) else np.array([0.0])

            feats.append(float(np.mean(obs_vals)))
            feats.append(float(np.std(obs_vals)))
            feats.append(float(np.min(obs_vals)))
            feats.append(float(np.max(obs_vals)))
            feats.append(float(v_ch[-1] - v_ch[0]))  # trajectory slope / trend
            feats.append(float(1.0 - np.mean(m_ch)))  # missingness rate

        # ECG summary features
        C_e, T_e = ecg.shape
        for c in range(C_e):
            e_ch = ecg[c]
            rms = np.sqrt(np.mean(e_ch ** 2) + 1e-8)
            p2p = np.ptp(e_ch)
            feats.append(float(rms))
            feats.append(float(p2p))
            feats.append(float(np.std(e_ch)))

        return np.array(feats, dtype=np.float32)

    def fit(self, samples: List[Dict[str, Any]]):
        X = [self.extract_features(s["ecg"], s["vitals"], s["mask"]) for s in samples]
        y = [s["label"] for s in samples]
        X = np.stack(X, axis=0)
        y = np.array(y, dtype=np.int32)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        logger.info(f"Fitted Logistic Regression baseline on {len(samples)} samples ({X.shape[1]} features).")

    def predict_proba(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Logistic Regression model is not fitted yet.")
        X = [self.extract_features(s["ecg"], s["vitals"], s["mask"]) for s in samples]
        X = np.stack(X, axis=0)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return probs
