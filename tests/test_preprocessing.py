import pytest
import numpy as np
from src.ecg_risk.data.preprocessing import ECGSignalPreprocessor, VitalSignsPreprocessor


def test_ecg_preprocessor_filtering_and_sqi():
    fs = 250
    duration_sec = 10
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    # Simulated ECG lead: 1.2 Hz cardiac rhythm + 60 Hz line noise + 0.2 Hz baseline wander
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 60.0 * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t)
    ecg_2ch = np.stack([ecg, ecg * 0.8], axis=0)

    preprocessor = ECGSignalPreprocessor(target_sampling_rate=fs)
    proc_ecg, sqi_list = preprocessor.process_record(ecg_2ch, orig_fs=fs)

    assert proc_ecg.shape == ecg_2ch.shape
    assert len(sqi_list) == 2
    assert all(0.0 <= sqi <= 1.0 for sqi in sqi_list)
    # Check that high frequency noise was filtered
    assert np.var(proc_ecg) < np.var(ecg_2ch)


def test_ecg_resampling():
    orig_fs = 500
    target_fs = 250
    t = np.linspace(0, 5, 5 * orig_fs)
    sig = np.sin(2 * np.pi * 2 * t)[np.newaxis, :]

    preprocessor = ECGSignalPreprocessor(target_sampling_rate=target_fs)
    proc, _ = preprocessor.process_record(sig, orig_fs=orig_fs)
    assert proc.shape == (1, 5 * target_fs)


def test_vital_signs_preprocessor_mask_and_imputation():
    channels = ["HR", "SBP", "DBP", "MAP", "SPO2", "RESP"]
    n_steps = 100
    vitals = np.ones((6, n_steps), dtype=np.float32) * 80.0
    # Inject physiological out of bounds outlier
    vitals[0, 10] = 999.0  # HR out of bounds
    # Inject missing segment
    vitals[1, 20:30] = np.nan

    preprocessor = VitalSignsPreprocessor(vital_channels=channels)
    proc, mask, missing_rate = preprocessor.process_record(vitals)

    assert proc.shape == (6, n_steps)
    assert mask.shape == (6, n_steps)
    assert not np.isnan(proc).any(), "Imputation must eliminate all NaNs"
    assert mask[0, 10] == 0.0, "Out of bounds value should be masked as missing"
    assert np.all(mask[1, 20:30] == 0.0), "NaN segment should be marked 0.0 in mask"
    assert missing_rate > 0.0


def test_train_set_normalization_isolation():
    """Verify normalizer fits ONLY on provided train samples without test leakage."""
    prep = ECGSignalPreprocessor()
    train_data = [np.random.normal(5.0, 2.0, (2, 1000)) for _ in range(5)]
    prep.fit_normalizer(train_data)

    assert prep.mean_ is not None
    assert np.isclose(prep.mean_[0, 0], 5.0, atol=0.5)

    test_data = np.random.normal(10.0, 1.0, (2, 500))
    norm_test = prep.normalize(test_data)
    # Normalized test data should be centered around (10 - 5) / 2 = 2.5, proving training stats were used
    assert np.isclose(np.mean(norm_test), 2.5, atol=0.5)
