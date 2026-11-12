import pytest
import numpy as np
from src.ecg_risk.data.labels import PatientLevelSplitter, LabelBuilder
from src.ecg_risk.data.alignment import SynchronizedWindowAligner


def test_leak_free_patient_level_splitter():
    patient_records = {}
    for i in range(100):
        pid = f"PATIENT_{i:03d}"
        patient_records[pid] = {
            "has_positive_event": (i % 3 == 0),
        }

    splitter = PatientLevelSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    train_pids, val_pids, test_pids = splitter.split_patients(patient_records)

    # Check partition sizes
    assert len(train_pids) == 70
    assert len(val_pids) == 15
    assert len(test_pids) == 15

    # CRITICAL LEAKAGE TEST: Zero intersection between all split pairs
    s_train = set(train_pids)
    s_val = set(val_pids)
    s_test = set(test_pids)

    assert len(s_train.intersection(s_val)) == 0, "Leakage between train and val!"
    assert len(s_train.intersection(s_test)) == 0, "Leakage between train and test!"
    assert len(s_val.intersection(s_test)) == 0, "Leakage between val and test!"


def test_label_builder_eligibility_and_exclusions():
    builder = LabelBuilder()

    # Eligible case
    ok, msg = builder.validate_record_eligibility(
        has_adequate_duration=True,
        has_verified_ecg=True,
        has_verified_vitals=True,
        is_pre_existing_cardiogenic_shock=False,
    )
    assert ok is True
    assert msg == "Eligible"

    # Pre-existing cardiogenic shock exclusion
    ok, msg = builder.validate_record_eligibility(
        has_adequate_duration=True,
        has_verified_ecg=True,
        has_verified_vitals=True,
        is_pre_existing_cardiogenic_shock=True,
    )
    assert ok is False
    assert "Excluded" in msg


def test_synchronized_window_alignment():
    ecg = np.zeros((2, 250 * 1800))
    vitals = np.zeros((6, 1800))
    mask = np.ones((6, 1800))
    # Event occurs at 1200 seconds (inside horizon after first window ending at 600s)
    event_ts = [1200.0]

    aligner = SynchronizedWindowAligner(
        context_window_sec=600,
        prediction_horizon_sec=3600,
        ecg_fs=250,
        vital_fs=1,
    )
    windows = aligner.extract_windows(
        ecg_signal=ecg,
        vital_signal=vitals,
        vital_mask=mask,
        event_timestamps_sec=event_ts,
        recording_duration_sec=1800,
        stride_sec=600,
    )

    assert len(windows) > 0
    # First window ends at 600s, event is at 1200s (600 <= 1200 <= 4200), so label should be 1.0
    assert windows[0]["label"] == 1.0
    assert windows[0]["ecg"].shape == (2, 250 * 600)
    assert windows[0]["vitals"].shape == (6, 600)
