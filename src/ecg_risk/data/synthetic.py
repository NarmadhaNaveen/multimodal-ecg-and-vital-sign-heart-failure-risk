
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SyntheticCohortGenerator:
    """
    Generate explicitly synthetic ECG and vital-sign demonstration data.

    Vital channels:
        HR, SBP, DBP, MAP, SPO2, RESP

    ECG channels:
        Lead II and Lead V

    IMPORTANT:
        This generator is for pipeline testing only. Its synthetic event
        labels must not be interpreted as clinically verified outcomes.
    """

    VITAL_NAMES = ["HR", "SBP", "DBP", "MAP", "SPO2", "RESP"]

    def __init__(
        self,
        num_patients: int = 50,
        context_window_sec: int = 600,
        prediction_horizon_sec: int = 14400,
        ecg_fs: int = 250,
        vital_fs: int = 1,
        seed: int = 42,
    ):
        if num_patients < 1:
            raise ValueError("num_patients must be at least 1")
        if context_window_sec <= 0 or prediction_horizon_sec <= 0:
            raise ValueError("Context and prediction horizon must be positive")
        if ecg_fs <= 0 or vital_fs <= 0:
            raise ValueError("Sampling rates must be positive")

        self.num_patients = num_patients
        self.context_window_sec = context_window_sec
        self.prediction_horizon_sec = prediction_horizon_sec
        self.ecg_fs = ecg_fs
        self.vital_fs = vital_fs
        self.seed = seed
        self._current_event_time_sec: Optional[float] = None

    def _generate_ecg_beat(
        self,
        hr_bpm: float,
        is_pathological: bool,
        lead: int = 0,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Generate one ECG beat using Gaussian waveform components."""
        hr_bpm = float(np.clip(hr_bpm, 30.0, 220.0))
        beat_len = max(2, int(self.ecg_fs * 60.0 / hr_bpm))
        t = np.linspace(-0.4, 0.6, beat_len)

        if not is_pathological:
            params = [
                (0.15, 0.04, -0.20),
                (-0.15, 0.02, -0.05),
                (1.20, 0.03, 0.00),
                (-0.25, 0.03, 0.05),
                (0.30, 0.08, 0.25),
            ]
        else:
            params = [
                (0.12, 0.05, -0.22),
                (-0.25, 0.03, -0.07),
                (0.90, 0.06, 0.00),
                (-0.35, 0.05, 0.08),
                (-0.40 if lead == 0 else 0.45, 0.12, 0.28),
            ]

        beat = np.zeros_like(t)
        for amplitude, width, center in params:
            beat += amplitude * np.exp(
                -((t - center) ** 2) / (2.0 * width**2)
            )

        if is_pathological:
            st_region = (t > 0.06) & (t < 0.22)
            beat[st_region] += 0.15 if lead == 0 else -0.15

        return beat

    def generate_patient_ecg(
        self,
        duration_sec: float,
        hr_series: np.ndarray,
        is_failing: bool,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Generate two ECG leads.

        The event-dependent morphology changes only at or after the
        configured synthetic event time. Before that time, both classes
        use the same ECG morphology generation rule.
        """
        if duration_sec <= 0 or len(hr_series) == 0:
            raise ValueError("Duration and HR series must be non-empty")

        total_samples = int(round(duration_sec * self.ecg_fs))
        ecg_signals = np.zeros((2, total_samples), dtype=np.float32)
        event_time = self._current_event_time_sec

        for lead in range(2):
            trace = []
            current_sec = 0.0

            while current_sec < duration_sec:
                idx = min(
                    int(current_sec * self.vital_fs),
                    len(hr_series) - 1,
                )
                bpm = float(hr_series[idx])

                pathological = (
                    event_time is not None
                    and current_sec >= event_time
                )

                beat = self._generate_ecg_beat(
                    bpm,
                    pathological,
                    lead=lead,
                    rng=rng,
                )
                trace.extend(beat)
                current_sec += len(beat) / self.ecg_fs

            signal = np.asarray(
                trace[:total_samples], dtype=np.float32
            )

            t_axis = np.arange(total_samples, dtype=np.float32) / self.ecg_fs
            signal += (
                0.15
                * np.sin(
                    2 * np.pi * 0.25 * t_axis
                    + rng.uniform(0, 2 * np.pi)
                )
            )
            signal += rng.normal(
                0, 0.04, total_samples
            ).astype(np.float32)

            ecg_signals[lead] = signal

        return ecg_signals

    def generate_patient_vitals(
        self,
        duration_sec: float,
        is_failing: bool,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate six synthetic vital channels and a missingness mask.

        The deterioration trajectory begins at the synthetic event time.
        Pre-event vital generation uses a stable baseline for both classes.
        """
        n_steps = int(round(duration_sec * self.vital_fs))
        if n_steps <= 0:
            raise ValueError("Duration must produce at least one vital sample")

        times = np.arange(n_steps, dtype=np.float64) / self.vital_fs

        base_hr = rng.uniform(68, 82)
        base_sbp = rng.uniform(115, 130)
        base_dbp = rng.uniform(70, 82)
        base_spo2 = rng.uniform(96, 99)
        base_resp = rng.uniform(14, 18)

        # Shared stable baseline generation for both outcome classes.
        hr = (
            base_hr
            + 4 * np.sin(2 * np.pi * 0.05 * times)
            + rng.normal(0, 1.5, n_steps)
        )
        sbp = (
            base_sbp
            + 5 * np.sin(2 * np.pi * 0.03 * times)
            + rng.normal(0, 2.0, n_steps)
        )
        dbp = (
            base_dbp
            + 3 * np.sin(2 * np.pi * 0.03 * times)
            + rng.normal(0, 1.5, n_steps)
        )
        spo2 = np.clip(
            base_spo2 + rng.normal(0, 0.5, n_steps),
            92,
            100,
        )
        resp = base_resp + rng.normal(0, 1.0, n_steps)

        event_time = self._current_event_time_sec

        if is_failing and event_time is not None:
            # Synthetic early-warning trend before the event.
            # This is an artificial simulation signal, not clinical evidence.
            context_end = float(self.context_window_sec)
            warning_start = max(
                0.0,
                min(event_time - 2.0 * 300.0, context_end - 300.0),
            )
            warning_end = min(event_time, context_end)
            warning_duration = max(warning_end - warning_start, 1.0)
            pre_event = (
                (times >= warning_start)
                & (times < warning_end)
            )

            warning_progress = np.zeros_like(times, dtype=np.float64)
            warning_progress[pre_event] = np.clip(
                (times[pre_event] - warning_start) / warning_duration,
                0.0,
                1.0,
            ) ** 1.5

            # Gradual changes during the final five minutes before the event.
            hr[pre_event] += 8.0 * warning_progress[pre_event]
            sbp[pre_event] -= 6.0 * warning_progress[pre_event]
            dbp[pre_event] -= 3.0 * warning_progress[pre_event]
            spo2[pre_event] = np.clip(
                spo2[pre_event] - 1.5 * warning_progress[pre_event],
                75,
                100,
            )
            resp[pre_event] += 2.0 * warning_progress[pre_event]

            # Stronger deterioration begins at the event boundary.
            elapsed = np.maximum(times - event_time, 0.0)
            ramp_duration = max(
                float(self.prediction_horizon_sec),
                1.0,
            )
            progression = np.clip(
                elapsed / ramp_duration, 0.0, 1.0
            ) ** 1.5

            after_event = times >= event_time
            hr[after_event] += (
                45.0 * progression[after_event]
                + rng.normal(0, 2.0, np.sum(after_event))
            )
            sbp[after_event] -= (
                45.0 * progression[after_event]
                - rng.normal(0, 3.0, np.sum(after_event))
            )
            dbp[after_event] -= (
                25.0 * progression[after_event]
                - rng.normal(0, 2.5, np.sum(after_event))
            )
            spo2[after_event] = np.clip(
                spo2[after_event]
                - 12.0 * progression[after_event]
                + rng.normal(0, 1.0, np.sum(after_event)),
                75,
                100,
            )
            resp[after_event] += (
                14.0 * progression[after_event]
                + rng.normal(0, 1.5, np.sum(after_event))
            )

        map_bp = (2.0 * dbp + sbp) / 3.0

        vitals = np.stack(
            [hr, sbp, dbp, map_bp, spo2, resp],
            axis=0,
        ).astype(np.float32)

        # 1 = observed; 0 = missing. Missing values remain NaN in vitals.
        mask = np.ones_like(vitals, dtype=np.float32)

        for channel in range(vitals.shape[0]):
            if rng.rand() < 0.6:
                max_burst = min(30, n_steps)
                burst_len = rng.randint(5, max_burst + 1) if max_burst >= 5 else max_burst
                if burst_len > 0:
                    burst_start = rng.randint(
                        0, max(1, n_steps - burst_len + 1)
                    )
                    vitals[
                        channel, burst_start:burst_start + burst_len
                    ] = np.nan
                    mask[
                        channel, burst_start:burst_start + burst_len
                    ] = 0.0

        return vitals, mask

    def generate_cohort(
        self,
        output_dir: str = "data/synthetic",
        pos_ratio: float = 0.35,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate synthetic patient records in memory.

        Every record has sufficient duration for one complete context plus
        prediction horizon. This does not establish real clinical outcomes.
        """
        if not 0.0 <= pos_ratio <= 1.0:
            raise ValueError("pos_ratio must be between 0 and 1")

        os.makedirs(output_dir, exist_ok=True)
        rng = np.random.RandomState(self.seed)
        all_patient_records: Dict[str, Dict[str, Any]] = {}

        duration_sec = (
            self.context_window_sec
            + self.prediction_horizon_sec
            + 1
        )

        logger.info(
            "Generating %d synthetic demonstration patients; duration=%d sec",
            self.num_patients,
            duration_sec,
        )

        for p_idx in range(self.num_patients):
            pid = f"SYNTH_PATIENT_{p_idx + 1:04d}"
            is_failing = bool(rng.rand() < pos_ratio)

            event_time = (
                self.context_window_sec + 300.0
                if is_failing
                else None
            )
            self._current_event_time_sec = event_time

            vitals, mask = self.generate_patient_vitals(
                duration_sec, is_failing, rng
            )
            hr_series = np.nan_to_num(
                vitals[0].copy(),
                nan=80.0,
                posinf=80.0,
                neginf=80.0,
            )

            ecg = self.generate_patient_ecg(
                duration_sec,
                hr_series,
                is_failing,
                rng,
            )

            all_patient_records[pid] = {
                "patient_id": pid,
                "has_positive_event": is_failing,
                "ecg": ecg,
                "vitals": vitals,
                "mask": mask,
                "event_ts": [event_time] if event_time is not None else [],
                "duration_sec": duration_sec,
                "synthetic_demo": True,
                "label_source": "synthetic_generator_not_clinically_verified",
                "vital_names": np.asarray(self.VITAL_NAMES),
                "sampling_rate_ecg": self.ecg_fs,
                "sampling_rate_vitals": self.vital_fs,
            }

        return all_patient_records
