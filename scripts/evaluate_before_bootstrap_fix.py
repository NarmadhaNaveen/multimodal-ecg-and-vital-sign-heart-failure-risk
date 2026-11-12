import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ecg_risk.config import ProjectConfig, ModelConfig
from src.ecg_risk.models.multimodal import MultimodalHeartFailureRiskModel
from src.ecg_risk.data.dataset import MultimodalECGDataset, multimodal_collate_fn
from src.ecg_risk.evaluation.metrics import evaluate_predictions, find_optimal_threshold
from src.ecg_risk.evaluation.calibration import TemperatureScaler
from src.ecg_risk.evaluation.confidence_intervals import bootstrap_metric_ci
from src.ecg_risk.evaluation.plots import generate_evaluation_plots
from scripts.train import load_npz_partition


@torch.no_grad()
def get_model_predictions(model, dataloader, device):
    """Computes logits, probabilities, and ground truth labels."""
    model.eval()
    all_logits = []
    all_probs = []
    all_targets = []
    all_pids = []

    for batch in dataloader:
        ecg = batch["ecg"].to(device, non_blocking=True)
        vitals = batch["vitals"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(ecg=ecg, vitals=vitals, mask=mask)

        all_logits.append(out["logits"].cpu().numpy())
        all_probs.append(out["probs"].cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_pids.extend(batch["patient_ids"])

    logits = np.concatenate(all_logits, axis=0).squeeze(-1)
    probs = np.concatenate(all_probs, axis=0).squeeze(-1)
    targets = np.concatenate(all_targets, axis=0).squeeze(-1)
    return logits, probs, targets, all_pids


def main():
    parser = argparse.ArgumentParser(description="Evaluate Heart-Failure Risk Model on held-out test data.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to trained checkpoint.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "mimic", "vitaldb"])
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Root directory containing dataset partitions.")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Directory to save evaluation plots and reports.")
    parser.add_argument("--bootstrap-ci", action="store_true", default=True, help="Compute patient-level bootstrap confidence intervals.")
    args = parser.parse_args()

    print("=" * 80)
    print("HELD-OUT EVALUATION & CLINICAL METRICS REPORT")
    print(f"CHECKPOINT: {args.checkpoint} | DATASET: {args.dataset.upper()}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    if not os.path.exists(args.checkpoint):
        print(f"CRITICAL: Checkpoint file not found: {args.checkpoint}")
        print("Please train a model first:\n  python scripts/train.py --dataset synthetic")
        sys.exit(1)

    # Load checkpoint & model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_raw = ckpt.get("config", {})
    if "model" in cfg_raw:
        model_cfg = cfg_raw["model"]
        if not hasattr(model_cfg, "architecture"):
            model_cfg = ModelConfig()
    else:
        model_cfg = ModelConfig()

    model = MultimodalHeartFailureRiskModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load validation and test cohorts
    proc_dir = os.path.join(args.data_dir, args.dataset)
    val_npz = os.path.join(proc_dir, "val.npz")
    test_npz = os.path.join(proc_dir, "test.npz")

    if not os.path.exists(val_npz) or not os.path.exists(test_npz):
        print(f"CRITICAL: Dataset partitions missing at {proc_dir}")
        sys.exit(1)

    val_samples = load_npz_partition(val_npz)
    test_samples = load_npz_partition(test_npz)

    val_loader = DataLoader(MultimodalECGDataset(val_samples), batch_size=16, shuffle=False, collate_fn=multimodal_collate_fn)
    test_loader = DataLoader(MultimodalECGDataset(test_samples), batch_size=16, shuffle=False, collate_fn=multimodal_collate_fn)

    # 1. Predictions on Validation Set
    val_logits, val_probs, val_targets, _ = get_model_predictions(model, val_loader, device)

    # Threshold Selection: STRICTLY on validation data only
    optimal_thresh = find_optimal_threshold(val_targets, val_probs, method="youden")
    print(f"\n[Validation Tuning] Selected Optimal Decision Threshold (Youden's J): {optimal_thresh:.3f}")

    # Temperature Scaling: Fit STRICTLY on validation data
    temp_scaler = TemperatureScaler()
    best_temp = temp_scaler.fit(val_logits, val_targets)
    print(f"[Validation Tuning] Temperature Scaling Parameter T: {best_temp:.3f}")

    # 2. Predictions on Held-Out Test Set
    test_logits, test_probs_raw, test_targets, test_pids = get_model_predictions(model, test_loader, device)
    test_probs_cal = temp_scaler.calibrate(test_logits)

    # 3. Comprehensive Metric Calculation on Test Set
    raw_metrics = evaluate_predictions(test_targets, test_probs_raw, threshold=optimal_thresh, logits=test_logits)
    cal_impact = temp_scaler.evaluate_calibration_impact(test_logits, test_targets)

    # 4. Bootstrap Confidence Intervals
    ci_results = {}
    if args.bootstrap_ci:
        print("\nComputing patient-level cluster bootstrap confidence intervals (n=500)...")
        ci_results = bootstrap_metric_ci(
            y_true=test_targets,
            y_prob=test_probs_cal,
            patient_ids=test_pids,
            n_bootstraps=500,
            threshold=optimal_thresh,
        )

    # Format Results Table
    table_rows = [
        ["Metric", "Measured Value", "95% Bootstrap CI", "Reference Benchmark"],
        ["AUROC", f"{raw_metrics['auroc']:.4f}",
         f"[{ci_results.get('auroc', {}).get('ci_lower', 0.0):.3f}, {ci_results.get('auroc', {}).get('ci_upper', 0.0):.3f}]" if ci_results else "N/A", "0.958"],
        ["AUPRC", f"{raw_metrics['auprc']:.4f}",
         f"[{ci_results.get('auprc', {}).get('ci_lower', 0.0):.3f}, {ci_results.get('auprc', {}).get('ci_upper', 0.0):.3f}]" if ci_results else "N/A", "0.931"],
        ["Accuracy", f"{raw_metrics['accuracy']:.4f}",
         f"[{ci_results.get('accuracy', {}).get('ci_lower', 0.0):.3f}, {ci_results.get('accuracy', {}).get('ci_upper', 0.0):.3f}]" if ci_results else "N/A", "0.956"],
        ["Sensitivity (Recall)", f"{raw_metrics['sensitivity']:.4f}",
         f"[{ci_results.get('sensitivity', {}).get('ci_lower', 0.0):.3f}, {ci_results.get('sensitivity', {}).get('ci_upper', 0.0):.3f}]" if ci_results else "N/A", "Target > 0.90"],
        ["Specificity", f"{raw_metrics['specificity']:.4f}",
         f"[{ci_results.get('specificity', {}).get('ci_lower', 0.0):.3f}, {ci_results.get('specificity', {}).get('ci_upper', 0.0):.3f}]" if ci_results else "N/A", "Target > 0.90"],
        ["F1-Score", f"{raw_metrics['f1_score']:.4f}",
         f"[{ci_results.get('f1_score', {}).get('ci_lower', 0.0):.3f}, {ci_results.get('f1_score', {}).get('ci_upper', 0.0):.3f}]" if ci_results else "N/A", "0.936"],
        ["ECE (Pre-Calibration)", f"{cal_impact['ece_before']:.4f}", "N/A", "0.031 - 0.036"],
        ["ECE (Post-Calibration)", f"{cal_impact['ece_after']:.4f}",
         f"[{ci_results.get('ece', {}).get('ci_lower', 0.0):.3f}, {ci_results.get('ece', {}).get('ci_upper', 0.0):.3f}]" if ci_results else "N/A", "0.019 - 0.022"],
        ["Brier Score (Pre-Cal)", f"{cal_impact['brier_before']:.4f}", "N/A", "N/A"],
        ["Brier Score (Post-Cal)", f"{cal_impact['brier_after']:.4f}", "N/A", "N/A"],
    ]

    print("\n--- HELD-OUT TEST COHORT PERFORMANCE ---")
    print(tabulate(table_rows, headers="firstrow", tablefmt="grid"))

    # Confusion matrix
    cm = raw_metrics["confusion_matrix"]
    print(f"\nConfusion Matrix (at validation-tuned threshold {optimal_thresh:.3f}):")
    print(f"  True Negatives (TN): {cm['tn']}  |  False Positives (FP): {cm['fp']}")
    print(f"  False Negatives (FN): {cm['fn']}  |  True Positives (TP):  {cm['tp']}")

    # 5. Generate and Save Plots
    os.makedirs(args.output_dir, exist_ok=True)
    plots = generate_evaluation_plots(
        y_true=test_targets,
        y_prob=test_probs_raw,
        output_dir=args.output_dir,
        prefix=args.dataset,
        y_prob_calibrated=test_probs_cal,
        threshold=optimal_thresh,
    )

    print("\nGenerated High-Resolution Evaluation Figures:")
    for name, path in plots.items():
        print(f"  {name.upper()}: {os.path.abspath(path)}")

    # 6. Export Results JSON
    summary_export = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "optimal_threshold": optimal_thresh,
        "temperature": best_temp,
        "metrics": raw_metrics,
        "calibration": cal_impact,
        "bootstrap_ci": ci_results,
        "num_test_samples": len(test_targets),
        "test_prevalence": float(np.mean(test_targets)),
    }
    json_path = os.path.join(args.output_dir, f"{args.dataset}_evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)

    print(f"\nFull evaluation report exported to: {os.path.abspath(json_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
