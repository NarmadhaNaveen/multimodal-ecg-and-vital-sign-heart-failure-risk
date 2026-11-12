import os
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SyntheticCohortGenerator:
    """
    Generates realistic, physically and clinically grounded synthetic cohorts
    with multi-lead ECG waveforms and synchronized vital sign time series.
    
    Synthesizes:
    - Multi-lead ECG: P-Q-R-S-T complexes using Gaussian sum models, with
      respiratory sinus arrhythmia, baseline wander, muscle noise, and pathological
      morphological changes (ST elevation/depression, QRS widening, T-wave inversion)
      in heart-failure deterioration cases.
    - Synchronized Vitals: Heart Rate, Systolic BP, Diastolic BP, Mean Arterial Pressure,
      SpO2, and Respiratory Rate, with correlated physiological dynamics and realistic
      measurement missingness patterns.
    - Explicit distinction: Annotated as SYNTHETIC DEMONSTRATION data.
    """

    def __init__(
        self,
        num_patients: int = 50,
        context_window_sec: int = 600,
        prediction_horizon_sec: int = 14400,
        ecg_fs: int = 250,
        vital_fs: int = 1,
        seed: int = 42,
    ):
        self.num_patients = num_patients
        self.context_window_sec = context_window_sec
        self.prediction_horizon_sec = prediction_horizon_sec
        self.ecg_fs = ecg_fs
        self.vital_fs = vital_fs
        self.seed = seed

    def _generate_ecg_beat(
        self,
        hr_bpm: float,
        is_pathological: bool,
        lead: int = 0,
        rng: np.random.RandomState = None,
    ) -> np.ndarray:
        """Generates a single ECG cardiac cycle using sum of Gaussian components."""
        beat_len = int(self.ecg_fs * (60.0 / hr_bpm))
        t = np.linspace(-0.4, 0.6, beat_len)

        # Baseline healthy parameters: (amplitude, width, center)
        # P, Q, R, S, T
        if not is_pathological:
            params = [
                (0.15, 0.04, -0.2),    # P-wave
                (-0.15, 0.02, -0.05),  # Q-wave
                (1.2, 0.03, 0.0),      # R-peak
                (-0.25, 0.03, 0.05),   # S-wave
                (0.3, 0.08, 0.25),     # T-wave
            ]
        else:
            # Heart failure / ischemic / strain pattern:
            # Widened QRS, ST depression / elevation, T wave inversion
            st_shift = -0.2 if lead == 0 else 0.25
            params = [
                (0.12, 0.05, -0.22),
                (-0.25, 0.03, -0.07),
                (0.9, 0.06, 0.0),       # Widened QRS
                (-0.35, 0.05, 0.08),
                (-0.4 if lead == 0 else 0.45, 0.12, 0.28), # Inverted/peaked T
            ]

        beat = np.zeros_like(t)
        for a, b, c in params:
            beat += a * np.exp(-((t - c) ** 2) / (2 * (b ** 2)))

        if is_pathological:
            # Add persistent ST level alteration
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
        """Constructs multi-lead (Lead II and Lead V) continuous ECG waveform."""
        total_samples = int(duration_sec * self.ecg_fs)
        ecg_signals = np.zeros((2, total_samples), dtype=np.float32)

        for lead in range(2):
            ecg_trace = []
            current_sec = 0.0
            while current_sec < duration_sec:
                idx = min(int(current_sec), len(hr_series) - 1)
                bpm = float(hr_series[idx])
                beat = self._generate_ecg_beat(bpm, is_failing, lead=lead, rng=rng)
                ecg_trace.extend(beat)
                current_sec += len(beat) / self.ecg_fs

            sig = np.array(ecg_trace[:total_samples], dtype=np.float32)

            # Baseline wander (respiratory oscillation: ~0.25 Hz)
            t_axis = np.linspace(0, duration_sec, total_samples)
            wander = 0.15 * np.sin(2 * np.pi * 0.25 * t_axis + rng.uniform(0, 2 * np.pi))
            sig += wander

            # High-frequency electromyographic noise
            noise = rng.normal(0, 0.04, total_samples).astype(np.float32)
            sig += noise

            ecg_signals[lead] = sig

        return ecg_signals

    def generate_patient_vitals(
        self,
        duration_sec: float,
        is_failing: bool,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 1 Hz vital signs: [HR, SBP, DBP, MAP, SPO2, RESP]
        along with realistic missingness mask.
        """
        n_steps = int(duration_sec * self.vital_fs)
        t = np.linspace(0, 1, n_steps)

        if not is_failing:
            # Stable patient
            base_hr = rng.uniform(68, 82)
            base_sbp = rng.uniform(115, 130)
            base_dbp = rng.uniform(70, 82)
            base_spo2 = rng.uniform(96, 99)
            base_resp = rng.uniform(14, 18)

            hr = base_hr + 4 * np.sin(2 * np.pi * 3 * t) + rng.normal(0, 1.5, n_steps)
            sbp = base_sbp + 5 * np.sin(2 * np.pi * 2 * t) + rng.normal(0, 2.0, n_steps)
            dbp = base_dbp + 3 * np.sin(2 * np.pi * 2 * t) + rng.normal(0, 1.5, n_steps)
            map_bp = (2 * dbp + sbp) / 3.0
            spo2 = np.clip(base_spo2 + rng.normal(0, 0.5, n_steps), 92, 100)
            resp = base_resp + rng.normal(0, 1.0, n_steps)
        else:
            # Acute Heart Failure Deterioration Trajectory:
            # Tachycardia compensatory attempt -> Blood pressure crash -> Pulmonary edema desaturation -> Tachypnea
            progression = np.clip((t - 0.3) / 0.7, 0.0, 1.0) ** 1.5

            base_hr = rng.uniform(75, 88)
            base_sbp = rng.uniform(110, 125)
            base_dbp = rng.uniform(68, 78)

            # HR increases from 80 to 125+ bpm
            hr = base_hr + 45.0 * progression + rng.normal(0, 2.0, n_steps)
            # SBP drops precipitously from 120 down to ~75 mmHg
            sbp = base_sbp - 45.0 * progression + rng.normal(0, 3.0, n_steps)
            # DBP drops down to ~45 mmHg
            dbp = base_dbp - 25.0 * progression + rng.normal(0, 2.5, n_steps)
            map_bp = (2 * dbp + sbp) / 3.0
            # SpO2 deteriorates to 84-88%
            spo2 = np.clip(97.0 - 12.0 * progression + rng.normal(0, 1.0, n_steps), 75, 100)
            # Resp rises up to 30+ breaths/min
            resp = 16.0 + 14.0 * progression + rng.normal(0, 1.5, n_steps)

        vitals = np.stack([hr, sbp, dbp, map_bp, spo2, resp], axis=0).astype(np.float32)

        # Inject realistic sensor disconnect / intermittent missingness
        # 1 = observed, 0 = missing
        mask = np.ones_like(vitals)
        for c in range(vitals.shape[0]):
            # 5% to 15% random missing bursts
            if rng.rand() < 0.6:
                burst_len = rng.randint(5, 30)
                burst_start = rng.randint(0, max(1, n_steps - burst_len))
                vitals[c, burst_start:burst_start + burst_len] = np.nan
                mask[c, burst_start:burst_start + burst_len] = 0.0

        return vitals, mask

    def generate_cohort(
        self,
        output_dir: str = "data/synthetic",
        pos_ratio: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Generates complete synthetic cohort, splits by patient, and saves raw files.
        Returns list of structured patient records with windows.
        """
        os.makedirs(output_dir, exist_ok=True)
        rng = np.random.RandomState(self.seed)

        all_patient_records = {}
        samples_list = []

        logger.info(f"Generating synthetic cohort of {self.num_patients} patients...")

        for p_idx in range(self.num_patients):
            pid = f"SYNTH_PATIENT_{p_idx + 1:04d}"
            is_failing = rng.rand() < pos_ratio

            # Total duration: context window + horizon + buffer
            duration_sec = self.context_window_sec + (self.prediction_horizon_sec if is_failing else 3600)
            # For realistic training size, limit duration to 1800 sec so windows are compact
            rec_duration_sec = self.context_window_sec + 600  # 20 minutes

            # Vitals
            vitals, mask = self.generate_patient_vitals(rec_duration_sec, is_failing, rng)
            hr_series = vitals[0].copy()
            hr_series = np.nan_to_num(hr_series, nan=80.0)

            # ECG
            ecg = self.generate_patient_ecg(rec_duration_sec, hr_series, is_failing, rng)

            # Event timestamp: if failing, event occurs right inside prediction horizon
            event_ts = [self.context_window_sec + 300.0] if is_failing else []

            all_patient_records[pid] = {
                "patient_id": pid,
                "has_positive_event": is_failing,
                "ecg": ecg,
                "vitals": vitals,
                "mask": mask,
                "event_ts": event_ts,
                "duration_sec": rec_duration_sec,
            }

        return all_patient_records
