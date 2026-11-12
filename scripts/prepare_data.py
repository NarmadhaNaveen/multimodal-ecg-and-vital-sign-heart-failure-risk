
import os
import sys
import argparse

import numpy as np
from tabulate import tabulate

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.ecg_risk.data.synthetic import SyntheticCohortGenerator
from src.ecg_risk.data.alignment import SynchronizedWindowAligner
from src.ecg_risk.data.labels import PatientLevelSplitter
from src.ecg_risk.data.preprocessing import (
    ECGSignalPreprocessor,
    VitalSignsPreprocessor,
)
from src.ecg_risk.data.mimic_loader import MIMIC3WaveformAdapter
from src.ecg_risk.data.vitaldb_loader import VitalDBAdapter


VITAL_CHANNELS = ["HR", "SBP", "DBP", "MAP", "SPO2", "RESP"]


def validate_args(args):
    if args.num_patients < 3:
        raise ValueError("--num-patients must be at least 3.")

    if args.context_window <= 0 or args.prediction_horizon <= 0:
        raise ValueError(
            "Context window and prediction horizon must be positive."
        )

    if args.normalization_stats and not os.path.isfile(
        args.normalization_stats
    ):
        raise FileNotFoundError(
            f"Normalization statistics not found: "
            f"{args.normalization_stats}"
        )


def validate_samples(samples, split_name):
    if not samples:
        raise ValueError(
            f"{split_name} partition has no windows. Increase "
            "--num-patients or adjust the context window, "
            "prediction horizon, and event settings."
        )

    for index, sample in enumerate(samples):
        for key in ("ecg", "vitals", "mask", "label"):
            if key not in sample:
                raise ValueError(
                    f"{split_name} sample {index} is missing '{key}'."
                )

        ecg = np.asarray(sample["ecg"])
        vitals = np.asarray(sample["vitals"])
        mask = np.asarray(sample["mask"])

        if ecg.ndim != 2 or ecg.shape[0] != 2:
            raise ValueError(
                f"{split_name} sample {index}: expected ECG shape "
                f"(2, time), got {ecg.shape}."
            )

        if vitals.ndim != 2 or vitals.shape[0] != len(VITAL_CHANNELS):
            raise ValueError(
                f"{split_name} sample {index}: expected vitals shape "
                f"({len(VITAL_CHANNELS)}, time), got {vitals.shape}."
            )

        if mask.shape != vitals.shape:
            raise ValueError(
                f"{split_name} sample {index}: mask shape {mask.shape} "
                f"does not match vitals shape {vitals.shape}."
            )

        if not np.isfinite(ecg).all():
            raise ValueError(
                f"{split_name} sample {index} contains non-finite ECG values."
            )

        observed = mask.astype(bool)
        if not np.isfinite(vitals[observed]).all():
            raise ValueError(
                f"{split_name} sample {index} contains non-finite "
                "observed vitals."
            )

        if sample["label"] not in (0, 0.0, 1, 1.0):
            raise ValueError(
                f"{split_name} sample {index} has invalid binary label "
                f"{sample['label']}."
            )


def save_partition(samples, out_dir, filename):
    if not samples:
        raise ValueError(
            f"Refusing to save empty partition: {filename}"
        )

    data = {
        "ecg": np.stack(
            [s["ecg"] for s in samples]
        ).astype(np.float32),
        "vitals": np.stack(
            [s["vitals"] for s in samples]
        ).astype(np.float32),
        "mask": np.stack(
            [s["mask"] for s in samples]
        ).astype(np.float32),
        "labels": np.asarray(
            [s["label"] for s in samples],
            dtype=np.float32,
        ),
        "patient_ids": np.asarray(
            [s["patient_id"] for s in samples]
        ),
    }

    for key in ("ecg", "vitals", "mask", "labels"):
        if not np.isfinite(data[key]).all():
            raise ValueError(
                f"Non-finite values found in output field '{key}'."
            )

    path = os.path.join(out_dir, filename)
    np.savez_compressed(path, **data)
    return path


def load_normalization_stats(stats_path, ecg_prep, vital_prep):
    """Load and validate previously fitted normalization statistics."""
    required = {
        "ecg_mean",
        "ecg_std",
        "vital_mean",
        "vital_std",
    }

    with np.load(stats_path, allow_pickle=False) as stats:
        missing = required.difference(stats.files)
        if missing:
            raise ValueError(
                f"Normalization file is missing keys: {sorted(missing)}"
            )

        ecg_mean = stats["ecg_mean"].copy()
        ecg_std = stats["ecg_std"].copy()
        vital_mean = stats["vital_mean"].copy()
        vital_std = stats["vital_std"].copy()

    expected_shapes = {
        "ecg_mean": (2, 1),
        "ecg_std": (2, 1),
        "vital_mean": (len(VITAL_CHANNELS), 1),
        "vital_std": (len(VITAL_CHANNELS), 1),
    }

    arrays = {
        "ecg_mean": ecg_mean,
        "ecg_std": ecg_std,
        "vital_mean": vital_mean,
        "vital_std": vital_std,
    }

    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(
                f"{name} has shape {array.shape}; "
                f"expected {expected_shapes[name]}."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} contains non-finite values."
            )

    if np.any(ecg_std <= 0) or np.any(vital_std <= 0):
        raise ValueError(
            "Normalization standard deviations must be positive."
        )

    ecg_prep.mean_ = ecg_mean
    ecg_prep.std_ = ecg_std
    vital_prep.mean_ = vital_mean
    vital_prep.std_ = vital_std

    print(f"Loaded fixed normalization statistics: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, synchronize, split, and normalize biomedical cohorts."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "mimic", "vitaldb"],
    )
    parser.add_argument("--num-patients", type=int, default=50)
    parser.add_argument("--context-window", type=int, default=600)
    parser.add_argument("--prediction-horizon", type=int, default=14400)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalization-stats",
        type=str,
        default=None,
        help=(
            "Path to an existing normalization_stats.npz file. "
            "If supplied, reuse its statistics instead of fitting new ones."
        ),
    )

    args = parser.parse_args()

    try:
        validate_args(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print("=" * 80)
    print(
        f"DATASET PREPARATION & PREPROCESSING PIPELINE: "
        f"{args.dataset.upper()}"
    )
    print("=" * 80)

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # These adapters currently check availability only.
    if args.dataset == "mimic":
        status = MIMIC3WaveformAdapter().check_access_and_records()
        print("MIMIC-III Waveform Matched Local Status:")
        print(f"  Records found: {status['record_count']}")

        if not status["available"]:
            print(status["instructions"])
            print("Use --dataset synthetic to test the pipeline.")
            return

        print(
            "MIMIC-III access is available, but this script does not yet "
            "implement conversion of those records into training windows."
        )
        return

    if args.dataset == "vitaldb":
        status = VitalDBAdapter().check_access_and_records()
        print("VitalDB Local Status:")
        print(f"  Records found: {status['record_count']}")

        if not status["available"]:
            print(status["instructions"])
            print("Use --dataset synthetic to test the pipeline.")
            return

        print(
            "VitalDB access is available, but this script does not yet "
            "implement conversion of those records into training windows."
        )
        return

    # Generate a cohort using the requested seed.
    print(
        f"Generating synthetic cohort: {args.num_patients} patients "
        f"(seed={args.seed})..."
    )

    generator = SyntheticCohortGenerator(
        num_patients=args.num_patients,
        context_window_sec=args.context_window,
        prediction_horizon_sec=args.prediction_horizon,
        ecg_fs=250,
        vital_fs=1,
        seed=args.seed,
    )

    patient_records = generator.generate_cohort()

    if len(patient_records) < 3:
        raise ValueError(
            "At least three generated patients are needed for splitting."
        )

    splitter = PatientLevelSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
    )

    train_pids, val_pids, test_pids = splitter.split_patients(
        patient_records
    )

    if not train_pids or not val_pids or not test_pids:
        raise ValueError(
            "Patient-level splitting produced an empty partition. "
            "Increase --num-patients."
        )

    aligner = SynchronizedWindowAligner(
        context_window_sec=args.context_window,
        prediction_horizon_sec=args.prediction_horizon,
        ecg_fs=250,
        vital_fs=1,
    )

    def extract_split_samples(patient_ids):
        samples = []

        for pid in patient_ids:
            record = patient_records[pid]

            windows = aligner.extract_windows(
                ecg_signal=record["ecg"],
                vital_signal=record["vitals"],
                vital_mask=record["mask"],
                event_timestamps_sec=record["event_ts"],
                recording_duration_sec=record["duration_sec"],
                stride_sec=300,
            )

            for window_index, window in enumerate(windows):
                window["patient_id"] = pid
                window["window_idx"] = window_index
                samples.append(window)

        return samples

    print("Extracting synchronized windows...")

    train_samples = extract_split_samples(train_pids)
    val_samples = extract_split_samples(val_pids)
    test_samples = extract_split_samples(test_pids)

    validate_samples(train_samples, "Train")
    validate_samples(val_samples, "Validation")
    validate_samples(test_samples, "Test")

    ecg_prep = ECGSignalPreprocessor(target_sampling_rate=250)
    vital_prep = VitalSignsPreprocessor(
        vital_channels=VITAL_CHANNELS,
        target_sampling_rate=1,
    )

    # Process ECG before fitting or applying normalization.
    print("Filtering and resampling ECG...")

    for samples in (train_samples, val_samples, test_samples):
        for sample in samples:
            processed_ecg, sqi = ecg_prep.process_record(
                np.asarray(sample["ecg"], dtype=np.float64),
                orig_fs=250,
            )
            sample["ecg"] = processed_ecg
            sample["sqi"] = sqi

    if args.normalization_stats:
        # Reuse the original training-fitted statistics.
        load_normalization_stats(
            args.normalization_stats,
            ecg_prep,
            vital_prep,
        )
    else:
        # Fit statistics on training data only.
        print("Fitting normalization statistics on training data...")
        ecg_prep.fit_normalizer(
            [s["ecg"] for s in train_samples]
        )
        vital_prep.fit_normalizer(
            [s["vitals"] for s in train_samples]
        )

        stats_path = os.path.join(
            out_dir,
            "normalization_stats.npz",
        )
        np.savez_compressed(
            stats_path,
            ecg_mean=ecg_prep.mean_,
            ecg_std=ecg_prep.std_,
            vital_mean=vital_prep.mean_,
            vital_std=vital_prep.std_,
        )
        print(f"Saved fitted normalization statistics: {stats_path}")

    # Apply the same statistics to all three partitions.
    print("Applying normalization...")

    for samples in (train_samples, val_samples, test_samples):
        for sample in samples:
            sample["ecg"] = ecg_prep.normalize(
                sample["ecg"]
            ).astype(np.float32)

            processed_vitals, processed_mask, missing_rate = (
                vital_prep.process_record(sample["vitals"])
            )

            sample["vitals"] = processed_vitals.astype(np.float32)
            sample["mask"] = processed_mask.astype(np.float32)
            sample["missing_rate"] = missing_rate

    for name, samples in (
        ("Train", train_samples),
        ("Validation", val_samples),
        ("Test", test_samples),
    ):
        validate_samples(samples, name)

    train_path = save_partition(
        train_samples, out_dir, "train.npz"
    )
    val_path = save_partition(
        val_samples, out_dir, "val.npz"
    )
    test_path = save_partition(
        test_samples, out_dir, "test.npz"
    )

    # Save one processed example for inspection.
    sample_rec_path = os.path.join(out_dir, "sample_record.npz")
    sample = test_samples[0]

    np.savez_compressed(
        sample_rec_path,
        ecg=sample["ecg"],
        vitals=sample["vitals"],
        mask=sample["mask"],
        patient_id=sample["patient_id"],
        label=sample["label"],
    )

    def summarize(name, pids, samples):
        positive = sum(
            float(s["label"]) == 1.0 for s in samples
        )
        prevalence = positive / len(samples)

        return [
            name,
            len(pids),
            len(samples),
            positive,
            f"{prevalence:.1%}",
        ]

    total_samples = train_samples + val_samples + test_samples
    total_positive = sum(
        float(s["label"]) == 1.0 for s in total_samples
    )

    rows = [
        [
            "Partition",
            "Patients",
            "Windows",
            "Positive Labels",
            "Prevalence",
        ],
        summarize("Train", train_pids, train_samples),
        summarize("Validation", val_pids, val_samples),
        summarize("Test", test_pids, test_samples),
        [
            "Total",
            len(patient_records),
            len(total_samples),
            total_positive,
            f"{total_positive / len(total_samples):.1%}",
        ],
    ]

    print(tabulate(rows[1:], headers=rows[0], tablefmt="grid"))

    print("\nFiles saved:")
    print(f"  Train:  {train_path}")
    print(f"  Val:    {val_path}")
    print(f"  Test:   {test_path}")
    print(f"  Sample: {sample_rec_path}")
    print("=" * 80)
    print("Data preparation completed successfully.")
    print(
        "Reminder: synthetic data is for software testing, "
        "not clinical validation."
    )


if __name__ == "__main__":
    main()
