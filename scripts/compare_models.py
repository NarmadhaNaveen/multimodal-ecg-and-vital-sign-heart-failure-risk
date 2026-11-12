import os
import sys
import argparse
import json
import numpy as np
import torch
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ecg_risk.config import ProjectConfig, ModelConfig
from src.ecg_risk.models.multimodal import MultimodalHeartFailureRiskModel
from src.ecg_risk.models.baselines import LogisticRegressionEngineeredBaseline
from src.ecg_risk.data.dataset import MultimodalECGDataset, multimodal_collate_fn
from src.ecg_risk.training.train import Trainer
from src.ecg_risk.evaluation.metrics import evaluate_predictions
from scripts.train import load_npz_partition


def run_baseline_logistic(train_samples, test_samples):
    """Trains and evaluates the feature-engineered Logistic Regression baseline."""
    lr_baseline = LogisticRegressionEngineeredBaseline()
    lr_baseline.fit(train_samples)
    probs = lr_baseline.predict_proba(test_samples)
    y_true = np.array([s["label"] for s in test_samples], dtype=int)
    metrics = evaluate_predictions(y_true, probs)
    return metrics


def run_ablation_model(architecture_name: str, config: ProjectConfig, train_samples, val_samples, test_samples, device):
    """Trains and evaluates a specific neural ablation model."""
    print(f"\n--- Training Ablation: {architecture_name} ---")
    cfg = ProjectConfig.from_yaml("configs/default.yaml")
    cfg.model.architecture = architecture_name
    cfg.training.max_epochs = 5  # Compact epochs for benchmark comparison
    cfg.training.batch_size = 8

    train_loader = torch.utils.data.DataLoader(
        MultimodalECGDataset(train_samples), batch_size=cfg.training.batch_size, shuffle=True, collate_fn=multimodal_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        MultimodalECGDataset(val_samples), batch_size=cfg.evaluation.batch_size, shuffle=False, collate_fn=multimodal_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        MultimodalECGDataset(test_samples), batch_size=cfg.evaluation.batch_size, shuffle=False, collate_fn=multimodal_collate_fn
    )

    trainer = Trainer(cfg, checkpoint_dir=f"checkpoints/ablation_{architecture_name}", output_dir=f"outputs/ablation_{architecture_name}")
    trainer.fit(train_loader, val_loader)

    # Evaluate on held-out test cohort
    metrics, probs, targets = trainer.evaluate(test_loader)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run and compare baseline models and architectural ablations.")
    parser.add_argument("--results-dir", type=str, default="outputs", help="Directory containing experiment summaries.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "mimic", "vitaldb"])
    parser.add_argument("--run-all-ablations", action="store_true", default=True, help="Train all ablation variants.")
    args = parser.parse_args()

    print("=" * 80)
    print("MULTIMODAL MODEL ABLATIONS & BASELINE COMPARISON BENCHMARK")
    print(f"DATASET: {args.dataset.upper()} | RESULTS DIR: {args.results_dir}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset partitions
    proc_dir = os.path.join("data", "processed", args.dataset)
    train_npz = os.path.join(proc_dir, "train.npz")
    val_npz = os.path.join(proc_dir, "val.npz")
    test_npz = os.path.join(proc_dir, "test.npz")

    if not os.path.exists(train_npz) or not os.path.exists(test_npz):
        print(f"Dataset partitions not found at: {proc_dir}")
        print("Please run data preparation first:\n  python scripts/prepare_data.py --dataset synthetic")
        sys.exit(1)

    train_samples = load_npz_partition(train_npz)
    val_samples = load_npz_partition(val_npz)
    test_samples = load_npz_partition(test_npz)

    config = ProjectConfig.from_yaml("configs/default.yaml")

    comparison_results = {}

    # 1. Baseline: Logistic Regression on Engineered Features
    print("\n[1/6] Evaluating Baseline: Logistic Regression (Engineered Features)...")
    lr_metrics = run_baseline_logistic(train_samples, test_samples)
    comparison_results["Logistic Regression (Engineered)"] = lr_metrics

    # 2. Ablations: Single Modalities and Intermediate Architectures
    ablations = [
        ("ECG-only 1D-CNN", "ecg_only"),
        ("Vital-sign-only TCN", "vitals_only"),
        ("Late-Fusion Baseline", "late_fusion"),
        ("CNN + TCN (No Transformer)", "cnn_tcn_no_transformer"),
        ("Transformer (No Cross-Attention)", "transformer_no_cross_attn"),
        ("Full Proposed Model", "full"),
    ]

    for display_name, arch_key in ablations:
        try:
            m = run_ablation_model(arch_key, config, train_samples, val_samples, test_samples, device)
            comparison_results[display_name] = m
        except Exception as e:
            print(f"Failed to evaluate ablation {display_name}: {e}")

    # Build Comparison Table
    table_rows = [
        ["Model / Architecture", "AUROC", "AUPRC", "Accuracy", "F1-Score", "ECE", "Target AUROC"]
    ]

    for model_name, m in comparison_results.items():
        table_rows.append([
            model_name,
            f"{m.get('auroc', 0.0):.4f}",
            f"{m.get('auprc', 0.0):.4f}",
            f"{m.get('accuracy', 0.0):.4f}",
            f"{m.get('f1_score', 0.0):.4f}",
            f"{m.get('ece', 0.0):.4f}",
            "0.958" if "Full" in model_name else "Baseline",
        ])

    print("\n" + "=" * 80)
    print("EXPERIMENTAL BENCHMARK & ABLATION COMPARISON (HELD-OUT TEST COHORT)")
    print("=" * 80)
    print(tabulate(table_rows, headers="firstrow", tablefmt="grid"))

    # Export comparison JSON
    out_path = os.path.join(args.results_dir, "model_comparison_benchmark.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\nModel comparison benchmark saved to: {os.path.abspath(out_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
