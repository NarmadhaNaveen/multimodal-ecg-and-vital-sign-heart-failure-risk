import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from ..config import ProjectConfig, ModelConfig
from ..models.multimodal import MultimodalHeartFailureRiskModel
from ..data.preprocessing import ECGSignalPreprocessor, VitalSignsPreprocessor
from ..evaluation.calibration import TemperatureScaler


class HeartFailureRiskPredictor:
    """
    Inference engine for multimodal ECG and vital-sign heart-failure risk stratification.
    Validates input signals, checks sampling rates and quality, runs forward inference,
    applies temperature scaling calibration, and reports risk tier and diagnostic warnings.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.checkpoint_path = checkpoint_path
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg_dict = ckpt.get("config", {})
        if "model" in cfg_dict:
            m_dict = cfg_dict["model"]
            if hasattr(m_dict, "__dict__"):
                self.model_cfg = m_dict
            else:
                self.model_cfg = ModelConfig()
        else:
            self.model_cfg = ModelConfig()

        self.model = MultimodalHeartFailureRiskModel(self.model_cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.temperature_scaler = TemperatureScaler()
        if "temperature" in ckpt:
            self.temperature_scaler.temperature = ckpt["temperature"]
            self.temperature_scaler.is_fitted = True

        self.ecg_preprocessor = ECGSignalPreprocessor()
        self.vital_preprocessor = VitalSignsPreprocessor(
            vital_channels=["HR", "SBP", "DBP", "MAP", "SPO2", "RESP"]
        )

    def validate_inputs(
        self,
        ecg_raw: np.ndarray,
        vitals_raw: np.ndarray,
        ecg_fs: float,
        vital_fs: float,
    ) -> List[str]:
        """Validates input signal dimensions, sampling frequencies, and values."""
        warnings = []
        if ecg_raw.ndim != 2:
            raise ValueError(f"Expected ECG shape (channels, samples), got {ecg_raw.shape}")
        if vitals_raw.ndim != 2:
            raise ValueError(f"Expected Vitals shape (channels, samples), got {vitals_raw.shape}")

        if ecg_fs < 50.0:
            warnings.append(f"Low ECG sampling rate ({ecg_fs} Hz). Minimum recommended is 125 Hz.")
        if np.isnan(ecg_raw).all():
            warnings.append("ECG signal contains all NaN values.")

        missing_vital_ratio = np.mean(np.isnan(vitals_raw))
        if missing_vital_ratio > 0.4:
            warnings.append(f"High vital sign missingness detected ({missing_vital_ratio:.1%}). Predictions may have wider uncertainty.")

        return warnings

    @torch.no_grad()
    def predict(
        self,
        ecg_raw: np.ndarray,
        vitals_raw: np.ndarray,
        ecg_fs: float = 250.0,
        vital_fs: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Runs complete inference pipeline for a single paired observation window.
        Input:
            ecg_raw: (C_ecg, T_ecg)
            vitals_raw: (C_vital, T_vital)
        Output:
            Risk score, classification, quality index, and diagnostic warnings.
        """
        warnings = self.validate_inputs(ecg_raw, vitals_raw, ecg_fs, vital_fs)

        # Preprocess ECG
        proc_ecg, sqi_list = self.ecg_preprocessor.process_record(ecg_raw, ecg_fs)
        min_sqi = min(sqi_list) if len(sqi_list) > 0 else 0.0
        if min_sqi < 0.4:
            warnings.append(f"Degraded ECG signal quality (SQI: {min_sqi:.2f}). Baseline noise or lead displacement suspected.")

        # Preprocess Vitals
        proc_vitals, mask, missing_rate = self.vital_preprocessor.process_record(vitals_raw)

        # Batch dimension
        t_ecg = torch.as_tensor(proc_ecg, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_vitals = torch.as_tensor(proc_vitals, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)

        out = self.model(ecg=t_ecg, vitals=t_vitals, mask=t_mask)
        logit = float(out["logits"].cpu().item())
        raw_prob = float(out["probs"].cpu().item())

        # Calibrated probability
        if self.temperature_scaler.is_fitted:
            cal_prob = float(self.temperature_scaler.calibrate(np.array([logit]))[0])
        else:
            cal_prob = raw_prob

        # Clinical Risk Stratification
        risk_tier = "High Risk" if cal_prob >= self.threshold else "Low / Standard Risk"

        return {
            "risk_score_raw": raw_prob,
            "risk_score_calibrated": cal_prob,
            "logit": logit,
            "threshold": self.threshold,
            "risk_tier": risk_tier,
            "high_risk_flag": bool(cal_prob >= self.threshold),
            "signal_quality_sqi": sqi_list,
            "vital_missing_rate": missing_rate,
            "warnings": warnings,
            "disclaimer": (
                "RESEARCH ONLY: Output is an experimental algorithmic risk stratification estimate "
                "derived from temporal neural models. It is not an FDA-cleared diagnostic test or clinical judgment."
            ),
        }
