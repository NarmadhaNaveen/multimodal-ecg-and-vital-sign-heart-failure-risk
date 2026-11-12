import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class LabelBuilder:
    """
    Defines and documents clinically defensible heart-failure risk outcomes.
    
    Default Outcome Definition:
    - Primary Endpoint: Acute Decompensated Heart Failure (ADHF) or Acute Perioperative Heart Failure.
    - Clinical Event Criteria:
      1. Initiation of intravenous inotropic support (e.g. milrinone, dobutamine, epinephrine)
         or mechanical circulatory assist device.
      2. Requirement for urgent non-invasive or invasive positive pressure ventilation due to
         acute cardiogenic pulmonary edema.
      3. Sustained hemodynamic collapse (mean arterial pressure < 55 mmHg or cardiac index < 2.0 L/min/m2)
         attributed to acute pump failure.
      4. Validated ICD-9/10 acute heart failure decompensation diagnosis code matching admission episode.
    
    Negative Endpoint:
    - Hospitalized patient monitored for at least the prediction horizon without triggering any
      of the acute decompensation criteria, discharge with stable cardiac status.
    
    Exclusions:
    - Isolated transient arrhythmias without hemodynamic compromise.
    - Isolated hypoxemia from primary pulmonary etiology without pump failure.
    - Pre-existing persistent cardiogenic shock prior to window onset.
    - Recordings lacking sufficient horizon observation duration.
    """

    def __init__(self, endpoint_name: str = "acute_decompensated_heart_failure"):
        self.endpoint_name = endpoint_name

    def validate_record_eligibility(
        self,
        has_adequate_duration: bool,
        has_verified_ecg: bool,
        has_verified_vitals: bool,
        is_pre_existing_cardiogenic_shock: bool = False,
    ) -> Tuple[bool, str]:
        """Validates if record meets clinical study inclusion and exclusion criteria."""
        if not has_verified_ecg:
            return False, "Excluded: ECG channels missing or corrupted."
        if not has_verified_vitals:
            return False, "Excluded: Mandatory vital signs missing."
        if not has_adequate_duration:
            return False, "Excluded: Recording duration shorter than context window + horizon."
        if is_pre_existing_cardiogenic_shock:
            return False, "Excluded: Pre-existing persistent shock prior to context window."
        return True, "Eligible"


class PatientLevelSplitter:
    """
    Partitions datasets strictly at the PATIENT level.
    Ensures zero data leakage between train, validation, and test sets.
    All windows derived from patient P belong exclusively to one partition.
    """

    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_patients(
        self,
        patient_records: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Takes patient_records mapping patient_id -> dict with 'has_positive_window' (bool).
        Performs stratified patient-level splitting to maintain balanced outcome prevalence.
        Returns: (train_pids, val_pids, test_pids)
        """
        rng = np.random.RandomState(self.seed)

        # Stratify by whether patient ever experienced an outcome
        pos_pids = [pid for pid, info in patient_records.items() if info.get("has_positive_event", False)]
        neg_pids = [pid for pid, info in patient_records.items() if not info.get("has_positive_event", False)]

        rng.shuffle(pos_pids)
        rng.shuffle(neg_pids)

        def split_list(lst: List[str]) -> Tuple[List[str], List[str], List[str]]:
            n = len(lst)
            n_train = int(round(n * self.train_ratio))
            n_val = int(round(n * self.val_ratio))
            train = lst[:n_train]
            val = lst[n_train:n_train + n_val]
            test = lst[n_train + n_val:]
            return train, val, test

        pos_train, pos_val, pos_test = split_list(pos_pids)
        neg_train, neg_val, neg_test = split_list(neg_pids)

        train_pids = sorted(pos_train + neg_train)
        val_pids = sorted(pos_val + neg_val)
        test_pids = sorted(pos_test + neg_test)

        logger.info(
            f"Patient-level Split: Total Patients={len(patient_records)} | "
            f"Train={len(train_pids)} ({len(pos_train)} pos) | "
            f"Val={len(val_pids)} ({len(pos_val)} pos) | "
            f"Test={len(test_pids)} ({len(pos_test)} pos)"
        )

        # Verification check: zero overlap
        s_train, s_val, s_test = set(train_pids), set(val_pids), set(test_pids)
        assert len(s_train.intersection(s_val)) == 0, "Patient leakage detected between train and val!"
        assert len(s_train.intersection(s_test)) == 0, "Patient leakage detected between train and test!"
        assert len(s_val.intersection(s_test)) == 0, "Patient leakage detected between val and test!"

        return train_pids, val_pids, test_pids
