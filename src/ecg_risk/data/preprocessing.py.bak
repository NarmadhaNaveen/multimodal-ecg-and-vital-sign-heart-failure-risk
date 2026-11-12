import numpy as np
import scipy.signal as signal
from typing import Tuple, Optional, Dict, Any, List


class ECGSignalPreprocessor:
    """
    Biomedical signal processing pipeline for raw ECG waveforms.
    Includes Butterworth bandpass filtering, mains notch filtering,
    Signal Quality Index (SQI) estimation, resampling, and normalization.
    """

    def __init__(
        self,
        target_sampling_rate: int = 250,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        filter_order: int = 3,
        notch_freq: float = 60.0,
        notch_quality: float = 30.0,
    ):
        self.target_fs = target_sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.notch_freq = notch_freq
        self.notch_quality = notch_quality
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def butter_bandpass_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Applies zero-phase Butterworth bandpass filter along time axis."""
        nyq = 0.5 * fs
        low = max(0.01, self.lowcut / nyq)
        high = min(0.99, self.highcut / nyq)
        b, a = signal.butter(self.filter_order, [low, high], btype="bandpass")
        # data shape: (channels, time) or (time,)
        if data.ndim == 1:
            return signal.filtfilt(b, a, data)
        return signal.filtfilt(b, a, data, axis=-1)

    def notch_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Removes powerline interference (50/60 Hz)."""
        nyq = 0.5 * fs
        w0 = self.notch_freq / nyq
        if w0 >= 1.0 or w0 <= 0.0:
            return data
        b, a = signal.iirnotch(w0, self.notch_quality)
        if data.ndim == 1:
            return signal.filtfilt(b, a, data)
        return signal.filtfilt(b, a, data, axis=-1)

    def resample_signal(self, data: np.ndarray, orig_fs: float) -> np.ndarray:
        """Resamples signal to target sampling rate."""
        if orig_fs == self.target_fs:
            return data
        num_samples = int(round(data.shape[-1] * self.target_fs / orig_fs))
        if data.ndim == 1:
            return signal.resample(data, num_samples)
        return signal.resample(data, num_samples, axis=-1)

    def compute_sqi(self, ecg_lead: np.ndarray, fs: float) -> float:
        """
        Computes Signal Quality Index (0.0 to 1.0).
        Evaluates power in the QRS band (5-15 Hz) relative to total power,
        and penalizes extreme clipping/flatlines.
        """
        if len(ecg_lead) == 0 or np.all(ecg_lead == 0):
            return 0.0
        # Check flatline percentage
        diffs = np.diff(ecg_lead)
        flatline_ratio = np.mean(np.abs(diffs) < 1e-6)
        if flatline_ratio > 0.5:
            return float(max(0.0, 1.0 - flatline_ratio))

        # Power spectral density
        freqs, psd = signal.welch(ecg_lead, fs=fs, nperseg=min(len(ecg_lead), int(fs * 2)))
        total_power = np.sum(psd) + 1e-10
        qrs_mask = (freqs >= 5.0) & (freqs <= 15.0)
        qrs_power = np.sum(psd[qrs_mask])
        ratio = qrs_power / total_power
        # Normal QRS power ratio is typically 0.3 - 0.7
        sqi = float(np.clip(ratio * 2.0, 0.0, 1.0))
        return sqi

    def fit_normalizer(self, train_signals: List[np.ndarray]):
        """Fits normalizer statistics ONLY on training set data to prevent leakage."""
        concatenated = np.concatenate(train_signals, axis=-1)  # (C, total_time)
        self.mean_ = np.nanmean(concatenated, axis=-1, keepdims=True)
        self.std_ = np.nanstd(concatenated, axis=-1, keepdims=True) + 1e-6

    def normalize(self, signal_arr: np.ndarray) -> np.ndarray:
        """Applies learned normalization."""
        if self.mean_ is None or self.std_ is None:
            # Per-record fallback standard scaler
            m = np.nanmean(signal_arr, axis=-1, keepdims=True)
            s = np.nanstd(signal_arr, axis=-1, keepdims=True) + 1e-6
            return (signal_arr - m) / s
        return (signal_arr - self.mean_) / self.std_

    def process_record(
        self,
        ecg_raw: np.ndarray,
        orig_fs: float,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Full pipeline: filtering -> notch -> resampling -> clipping -> SQI calculation.
        Input ecg_raw: shape (C, T) or (T,)
        Returns: (processed_ecg, sqi_per_channel)
        """
        if ecg_raw.ndim == 1:
            ecg_raw = ecg_raw[np.newaxis, :]

        # Replace non-finite values
        ecg_clean = np.nan_to_num(ecg_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Filtering
        filtered = self.butter_bandpass_filter(ecg_clean, orig_fs)
        notched = self.notch_filter(filtered, orig_fs)

        # Resampling
        resampled = self.resample_signal(notched, orig_fs)

        # Clip extreme artifacts at 99.9th percentile
        p_high = np.percentile(np.abs(resampled), 99.9)
        if p_high > 0:
            resampled = np.clip(resampled, -p_high, p_high)

        # Compute SQI per channel
        sqi_list = [self.compute_sqi(resampled[c], self.target_fs) for c in range(resampled.shape[0])]

        return resampled, sqi_list


class VitalSignsPreprocessor:
    """
    Preprocessor for vital signs time series.
    Handles physiological bounds checking, explicit missingness mask generation,
    imputation, and training-statistics normalization.
    """

    # Clinically plausible ranges for ICU/OR vitals
    PHYSIOLOGICAL_BOUNDS = {
        "HR": (20.0, 240.0),
        "SBP": (30.0, 300.0),
        "DBP": (15.0, 200.0),
        "MAP": (20.0, 250.0),
        "SPO2": (40.0, 100.0),
        "RESP": (4.0, 60.0),
    }

    def __init__(
        self,
        vital_channels: List[str],
        target_sampling_rate: int = 1,
    ):
        self.vital_channels = vital_channels
        self.target_fs = target_sampling_rate
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit_normalizer(self, train_vitals: List[np.ndarray]):
        """Calculates mean and std across valid (non-missing) measurements in train set."""
        concatenated = np.concatenate(train_vitals, axis=-1)  # (C, total_time)
        self.mean_ = np.nanmean(concatenated, axis=-1, keepdims=True)
        self.std_ = np.nanstd(concatenated, axis=-1, keepdims=True) + 1e-6
        # Fallback if channel is all NaN
        self.mean_ = np.nan_to_num(self.mean_, nan=0.0)
        self.std_ = np.where(np.isnan(self.std_) | (self.std_ < 1e-6), 1.0, self.std_)

    def process_record(
        self,
        vitals_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Processes vital signs array of shape (num_channels, time_steps).
        Returns:
            processed_vitals: normalized imputed vitals array (num_channels, time_steps)
            missingness_mask: binary indicator mask (1 if observed, 0 if missing) (num_channels, time_steps)
            missing_rate: overall fraction of missing values
        """
        vitals = vitals_raw.copy()
        C, T = vitals.shape

        # Filter against physiological bounds
        for c, ch_name in enumerate(self.vital_channels):
            bounds = self.PHYSIOLOGICAL_BOUNDS.get(ch_name.upper())
            if bounds is not None:
                invalid_mask = (vitals[c] < bounds[0]) | (vitals[c] > bounds[1])
                vitals[c, invalid_mask] = np.nan

        # Missingness mask: 1 = valid observation, 0 = missing
        observed_mask = (~np.isnan(vitals)).astype(np.float32)
        missing_rate = float(1.0 - np.mean(observed_mask))

        # Forward-fill, then backward-fill, then fill remainder with median / zero
        imputed = np.zeros_like(vitals)
        for c in range(C):
            series = vitals[c]
            # If all missing, fill with 0.0
            if np.all(np.isnan(series)):
                imputed[c] = 0.0
                continue
            # Pandas-free forward/backward fill
            valid_idx = np.where(~np.isnan(series))[0]
            first_valid = valid_idx[0]
            last_valid = valid_idx[-1]

            filled = series.copy()
            # Forward fill
            for t in range(first_valid + 1, T):
                if np.isnan(filled[t]):
                    filled[t] = filled[t - 1]
            # Backward fill head
            filled[:first_valid] = filled[first_valid]
            imputed[c] = filled

        # Normalization
        if self.mean_ is not None and self.std_ is not None:
            normalized = (imputed - self.mean_) / self.std_
        else:
            m = np.nanmean(imputed, axis=-1, keepdims=True)
            s = np.nanstd(imputed, axis=-1, keepdims=True) + 1e-6
            normalized = (imputed - m) / s

        normalized = np.nan_to_num(normalized, nan=0.0)
        return normalized, observed_mask, missing_rate
