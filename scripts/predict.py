import os
import sys
import argparse
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ecg_risk.inference.predictor import HeartFailureRiskPredictor


def main():
    parser = argparse.ArgumentParser(description="Run Heart-Failure Risk Prediction Inference on an observation file.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to trained model checkpoint.")
    parser.add_argument("--input", type=str, required=True, help="Path to observation record (.npz or .npy).")
    parser.add_argument("--ecg-fs", type=float, default=250.0, help="Sampling frequency of input ECG.")
    parser.add_argument("--vital-fs", type=float, default=1.0, help="Sampling frequency of input vitals.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold.")
    args = parser.parse_args()

    print("=" * 80)
    print("MULTIMODAL HEART-FAILURE RISK STRATIFICATION: CLINICAL INFERENCE")
    print("=" * 80)

    if not os.path.exists(args.checkpoint):
        print(f"CRITICAL: Checkpoint not found: {args.checkpoint}")
        print("Please train a model first using:\n  python scripts/train.py --dataset synthetic")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"CRITICAL: Input observation record not found: {args.input}")
        sys.exit(1)

    # Load input data
    print(f"Loading observation record: {args.input}...")
    if args.input.endswith(".npz"):
        data = np.load(args.input)
        ecg = data["ecg"]
        vitals = data["vitals"]
        patient_id = str(data.get("patient_id", "ANONYMIZED_RECORD"))
        true_label = data.get("label", None)
    else:
        print("Supported format is .npz containing 'ecg' and 'vitals' arrays.")
        sys.exit(1)

    print(f"  Patient Identifier: {patient_id}")
    print(f"  ECG Shape:          {ecg.shape} (fs = {args.ecg_fs} Hz)")
    print(f"  Vitals Shape:       {vitals.shape} (fs = {args.vital_fs} Hz)")

    # Run inference
    predictor = HeartFailureRiskPredictor(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )
    result = predictor.predict(
        ecg_raw=ecg,
        vitals_raw=vitals,
        ecg_fs=args.ecg_fs,
        vital_fs=args.vital_fs,
    )

    # Format output table
    res_table = [
        ["Parameter", "Result", "Notes"],
        ["Calibrated Risk Probability", f"{result['risk_score_calibrated']:.1%}", "Temperature-scaled confidence"],
        ["Raw Model Score", f"{result['risk_score_raw']:.4f}", f"Logit: {result['logit']:.3f}"],
        ["Decision Threshold", f"{result['threshold']:.2f}", "Validation-tuned cutoff"],
        ["Stratification Tier", result['risk_tier'].upper(), "HIGH RISK" if result['high_risk_flag'] else "STANDARD"],
        ["Vital Missingness Rate", f"{result['vital_missing_rate']:.1%}", "Forward-filled + Masked"],
        ["ECG Signal Quality (SQI)", ", ".join(f"{s:.2f}" for s in result['signal_quality_sqi']), "QRS-band energy index (0-1)"],
    ]

    if true_label is not None:
        res_table.append(["Documented True Endpoint", f"{int(true_label)} ({'HF Event' if true_label==1 else 'No Event'})", "Ground Truth Annotation"])

    print("\n" + tabulate(res_table, headers="firstrow", tablefmt="grid"))

    if result["warnings"]:
        print("\nDIAGNOSTIC / SIGNAL WARNINGS:")
        for w in result["warnings"]:
            print(f"  [!] {w}")
    else:
        print("\nAll input signals passed biomedical quality integrity checks.")

    print("\n" + "-" * 80)
    print(f"DISCLAIMER: {result['disclaimer']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
