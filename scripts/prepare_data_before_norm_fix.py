import os
import sys
import argparse
import json
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ecg_risk.data.synthetic import SyntheticCohortGenerator
from src.ecg_risk.data.alignment import SynchronizedWindowAligner
from src.ecg_risk.data.labels import PatientLevelSplitter
from src.ecg_risk.data.preprocessing import ECGSignalPreprocessor, VitalSignsPreprocessor
from src.ecg_risk.data.mimic_loader import MIMIC3WaveformAdapter
from src.ecg_risk.data.vitaldb_loader import VitalDBAdapter

def main():
    parser = argparse.ArgumentParser(description="Prepare, synchronize, split, and normalize biomedical cohorts.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "mimic", "vitaldb"],
                        help="Dataset source adapter to run.")
    parser.add_argument("--num-patients", type=int, default=50, help="Number of synthetic patients to generate.")
    parser.add_argument("--context-window", type=int, default=600, help="Historical context window length in seconds.")
    parser.add_argument("--prediction-horizon", type=int, default=14400, help="Prediction horizon in seconds.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save processed splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for patient splitting.")
    args = parser.parse_args()

    print("=" * 80)
    print(f"DATASET PREPARATION & PREPROCESSING PIPELINE: {args.dataset.upper()}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == "mimic":
        adapter = MIMIC3WaveformAdapter()
        status = adapter.check_access_and_records()
        print(f"MIMIC-III Waveform Matched Local Status:")
        print(f"  Records found: {status['record_count']}")
        if not status["available"]:
            print("\n" + "=" * 80)
            print("ACCESS NOTICE: Genuine PhysioNet MIMIC-III Waveform Matched records not detected.")
            print(status["instructions"])
            print("=" * 80)
            print("To proceed with training without delay, please run with '--dataset synthetic'.")
            return

    elif args.dataset == "vitaldb":
        adapter = VitalDBAdapter()
        status = adapter.check_access_and_records()
        print(f"VitalDB Local Status:")
        print(f"  Records found: {status['record_count']}")
        if not status["available"]:
            print("\n" + "=" * 80)
            print("ACCESS NOTICE: Genuine VitalDB perioperative records not detected.")
            print(status["instructions"])
            print("=" * 80)
            print("To proceed with training without delay, please run with '--dataset synthetic'.")
            return

    elif args.dataset == "synthetic":
        print(f"Generating synthetic cohort: {args.num_patients} patients...")
        generator = SyntheticCohortGenerator(
            num_patients=args.num_patients,
            context_window_sec=args.context_window,
            prediction_horizon_sec=args.prediction_horizon,
            ecg_fs=250,
            vital_fs=1,
            seed=args.seed,
        )
        patient_records = generator.generate_cohort()

        # Split patients strictly at PATIENT level (0 data leakage)
        splitter = PatientLevelSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
        train_pids, val_pids, test_pids = splitter.split_patients(patient_records)

        # Synchronize and extract fixed context windows
        aligner = SynchronizedWindowAligner(
            context_window_sec=args.context_window,
            prediction_horizon_sec=args.prediction_horizon,
            ecg_fs=250,
            vital_fs=1,
        )

        def extract_split_samples(pids):
            split_samples = []
            for pid in pids:
                p_info = patient_records[pid]
                windows = aligner.extract_windows(
                    ecg_signal=p_info["ecg"],
                    vital_signal=p_info["vitals"],
                    vital_mask=p_info["mask"],
                    event_timestamps_sec=p_info["event_ts"],
                    recording_duration_sec=p_info["duration_sec"],
                    stride_sec=300,
                )
                for w_idx, w in enumerate(windows):
                    w["patient_id"] = pid
                    w["window_idx"] = w_idx
                    split_samples.append(w)
            return split_samples

        print("Aligning and extracting temporal observation windows...")
        train_samples = extract_split_samples(train_pids)
        val_samples = extract_split_samples(val_pids)
        test_samples = extract_split_samples(test_pids)

        # Fit preprocessors ONLY on training set to prevent normalization leakage
        print("Fitting normalizer statistics strictly on training cohort...")
        ecg_prep = ECGSignalPreprocessor(target_sampling_rate=250)
        train_ecg_list = [s["ecg"] for s in train_samples]
        ecg_prep.fit_normalizer(train_ecg_list)

        vital_prep = VitalSignsPreprocessor(
            vital_channels=["HR", "SBP", "DBP", "MAP", "SPO2", "RESP"],
            target_sampling_rate=1,
        )
        train_vital_list = [s["vitals"] for s in train_samples]
        vital_prep.fit_normalizer(train_vital_list)

        # Process each partition
        def apply_preprocessing(samples):
            for s in samples:
                proc_ecg, sqi = ecg_prep.process_record(s["ecg"], orig_fs=250)
                proc_vit, mask, miss_rate = vital_prep.process_record(s["vitals"])
                s["ecg"] = proc_ecg.astype(np.float32)
                s["vitals"] = proc_vit.astype(np.float32)
                s["mask"] = mask.astype(np.float32)
                s["sqi"] = sqi
                s["missing_rate"] = miss_rate

        apply_preprocessing(train_samples)
        apply_preprocessing(val_samples)
        apply_preprocessing(test_samples)

        # Save to disk
        def save_partition(samples, filename):
            save_path = os.path.join(out_dir, filename)
            data_dict = {
                "ecg": np.stack([s["ecg"] for s in samples]),
                "vitals": np.stack([s["vitals"] for s in samples]),
                "mask": np.stack([s["mask"] for s in samples]),
                "labels": np.array([s["label"] for s in samples], dtype=np.float32),
                "patient_ids": np.array([s["patient_id"] for s in samples]),
            }
            np.savez_compressed(save_path, **data_dict)
            return save_path

        train_path = save_partition(train_samples, "train.npz")
        val_path = save_partition(val_samples, "val.npz")
        test_path = save_partition(test_samples, "test.npz")

        # Save a single standalone sample record for scripts/predict.py
        sample_rec_path = os.path.join("data", "sample_record.npz")
        np.savez(
            sample_rec_path,
            ecg=test_samples[0]["ecg"],
            vitals=test_samples[0]["vitals"],
            patient_id=test_samples[0]["patient_id"],
            label=test_samples[0]["label"],
        )

        # Summary statistics
        train_pos = sum(1 for s in train_samples if s["label"] == 1.0)
        val_pos = sum(1 for s in val_samples if s["label"] == 1.0)
        test_pos = sum(1 for s in test_samples if s["label"] == 1.0)

        summary_table = [
            ["Partition", "Patients", "Windows", "Positive Labels", "Prevalence"],
            ["Train", len(train_pids), len(train_samples), train_pos, f"{train_pos/len(train_samples):.1%}"],
            ["Val", len(val_pids), len(val_samples), val_pos, f"{val_pos/len(val_samples):.1%}"],
            ["Test", len(test_pids), len(test_samples), test_pos, f"{test_pos/len(test_samples):.1%}"],
            ["Total", len(patient_records), len(train_samples)+len(val_samples)+len(test_samples),
             train_pos+val_pos+test_pos, f"{(train_pos+val_pos+test_pos)/(len(train_samples)+len(val_samples)+len(test_samples)):.1%}"]
        ]
        print(tabulate(summary_table, headers="firstrow", tablefmt="grid"))
        print(f"\nFiles saved:")
        print(f"  Train:  {train_path}")
        print(f"  Val:    {val_path}")
        print(f"  Test:   {test_path}")
        print(f"  Sample: {sample_rec_path}")
        print("=" * 80)
        print("Data preparation completed successfully.")

if __name__ == "__main__":
    main()
