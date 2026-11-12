import numpy as np
from typing import Any, Dict, List


class SynchronizedWindowAligner:
    """Extract paired ECG/vital context windows with valid future outcomes."""

    def __init__(
        self,
        context_window_sec: int = 600,
        prediction_horizon_sec: int = 14400,
        ecg_fs: int = 250,
        vital_fs: int = 1,
    ):
        self.context_window_sec = context_window_sec
        self.prediction_horizon_sec = prediction_horizon_sec
        self.ecg_fs = ecg_fs
        self.vital_fs = vital_fs

    def extract_windows(
        self,
        ecg_signal: np.ndarray,
        vital_signal: np.ndarray,
        vital_mask: np.ndarray,
        event_timestamps_sec: List[float],
        recording_duration_sec: float,
        stride_sec: int = 300,
    ) -> List[Dict[str, Any]]:
        windows = []

        if stride_sec <= 0:
            raise ValueError("stride_sec must be positive")

        events = sorted(float(t) for t in event_timestamps_sec)
        context = self.context_window_sec
        horizon = self.prediction_horizon_sec

        max_start = int(recording_duration_sec - context - horizon)
        if max_start < 0:
            return windows

        for t_start in range(0, max_start + 1, stride_sec):
            t_end = t_start + context
            horizon_end = t_end + horizon

            # Do not use context windows that begin at or after a prior event.
            if any(event < t_end for event in events):
                continue

            ecg_start = int(t_start * self.ecg_fs)
            ecg_end = int(t_end * self.ecg_fs)
            vital_start = int(t_start * self.vital_fs)
            vital_end = int(t_end * self.vital_fs)

            if (
                ecg_end > ecg_signal.shape[-1]
                or vital_end > vital_signal.shape[-1]
                or vital_end > vital_mask.shape[-1]
            ):
                continue

            ecg_slice = ecg_signal[..., ecg_start:ecg_end]
            vital_slice = vital_signal[..., vital_start:vital_end]
            mask_slice = vital_mask[..., vital_start:vital_end]

            if not (
                np.isfinite(ecg_slice).all()
                and np.isfinite(vital_slice).all()
                and np.isfinite(mask_slice).all()
            ):
                continue

            # A positive event must occur strictly after context and within horizon.
            positive = any(t_end <= event <= horizon_end for event in events)

            windows.append({
                "ecg": ecg_slice.astype(np.float32),
                "vitals": vital_slice.astype(np.float32),
                "mask": mask_slice.astype(np.float32),
                "label": float(positive),
                "t_start_sec": float(t_start),
                "t_end_sec": float(t_end),
            })

        return windows
