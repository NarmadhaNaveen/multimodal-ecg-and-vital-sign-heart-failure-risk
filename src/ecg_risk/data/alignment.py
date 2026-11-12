
import numpy as np
from typing import Any, Dict, List


class SynchronizedWindowAligner:
    """Extract synchronized context windows with observable future outcomes."""

    def __init__(
        self,
        context_window_sec: int = 600,
        prediction_horizon_sec: int = 14400,
        ecg_fs: int = 250,
        vital_fs: int = 1,
    ):
        if context_window_sec <= 0:
            raise ValueError("context_window_sec must be positive")
        if prediction_horizon_sec <= 0:
            raise ValueError("prediction_horizon_sec must be positive")
        if ecg_fs <= 0 or vital_fs <= 0:
            raise ValueError("Sampling frequencies must be positive")

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
        if stride_sec <= 0:
            raise ValueError("stride_sec must be positive")
        if recording_duration_sec < 0:
            raise ValueError("recording_duration_sec cannot be negative")

        ecg_signal = np.asarray(ecg_signal)
        vital_signal = np.asarray(vital_signal)
        vital_mask = np.asarray(vital_mask)

        if ecg_signal.ndim < 1:
            raise ValueError("ecg_signal must have at least one dimension")
        if vital_signal.shape != vital_mask.shape:
            raise ValueError("vital_signal and vital_mask must have the same shape")

        events = []
        for timestamp in event_timestamps_sec:
            event = float(timestamp)
            if not np.isfinite(event) or event < 0:
                raise ValueError(
                    "Event timestamps must be finite and non-negative"
                )
            events.append(event)
        events.sort()

        context = self.context_window_sec
        horizon = self.prediction_horizon_sec

        # Never trust the supplied duration beyond the available arrays.
        available_duration = min(
            float(recording_duration_sec),
            ecg_signal.shape[-1] / self.ecg_fs,
            vital_signal.shape[-1] / self.vital_fs,
        )

        max_start = int(np.floor(available_duration - context))
        if max_start < 0:
            return []

        windows = []

        for t_start in range(0, max_start + 1, stride_sec):
            t_end = t_start + context
            horizon_end = t_end + horizon

            # An event at or before context end is not a future prediction.
            if any(event <= t_end for event in events):
                continue

            # Only events actually observed by recording end can establish
            # a positive label.
            positive = any(
                t_end < event <= horizon_end
                and event <= available_duration
                for event in events
            )

            # A negative label requires complete follow-up through the horizon.
            if not positive and horizon_end > available_duration:
                continue

            ecg_start = int(round(t_start * self.ecg_fs))
            ecg_end = int(round(t_end * self.ecg_fs))
            vital_start = int(round(t_start * self.vital_fs))
            vital_end = int(round(t_end * self.vital_fs))

            ecg_slice = ecg_signal[..., ecg_start:ecg_end]
            vital_slice = vital_signal[..., vital_start:vital_end]
            mask_slice = vital_mask[..., vital_start:vital_end]

            expected_ecg = int(round(context * self.ecg_fs))
            expected_vitals = int(round(context * self.vital_fs))

            if (
                ecg_slice.shape[-1] != expected_ecg
                or vital_slice.shape[-1] != expected_vitals
                or mask_slice.shape[-1] != expected_vitals
            ):
                continue

            if not np.isfinite(ecg_slice).all():
                continue
            if not np.isfinite(mask_slice).all():
                continue

            # Mask > 0 means the vital observation is present.
            observed = mask_slice > 0
            if not np.isfinite(vital_slice[observed]).all():
                continue

            windows.append({
                "ecg": ecg_slice.astype(np.float32),
                "vitals": vital_slice.astype(np.float32),
                "mask": mask_slice.astype(np.float32),
                "label": float(positive),
                "t_start_sec": float(t_start),
                "t_end_sec": float(t_end),
            })

        return windows
