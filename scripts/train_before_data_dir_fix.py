import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ecg_risk.config import ProjectConfig
from src.ecg_risk.device import verify_gpu_or_raise, get_environment_info
from src.ecg_risk.data.dataset import MultimodalECGDataset, multimodal_collate_fn
from src.ecg_risk.training.train import Trainer


def load_npz_partition(npz_path: str):
    """Loads preprocessed .npz partition into list of sample dicts."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Partition file not found: {npz_path}")
    data = np.load(npz_path)
    ecg = data["ecg"]
    vitals = data["vitals"]
    mask = data["mask"]
    labels = data["labels"]
    patient_ids = data["patient_ids"]

    samples = []
    for i in range(len(labels)):
        samples.append({
            "ecg": ecg[i],
            "vitals": vitals[i],
            "mask": mask[i],
            "label": float(labels[i]),
            "patient_id": str(patient_ids[i]),
            "window_idx": i,
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal ECG & Vital-Sign Heart-Failure Risk Model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to experiment config YAML.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "mimic", "vitaldb"],
                        help="Dataset partition to train on.")
    parser.add_argument("--epochs", type=int, default=None, help="Override maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke-test training (1-2 epochs).")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save logs and metrics.")
    args = parser.parse_args()

    print("=" * 80)
    print("MULTIMODAL ECG & VITAL-SIGN HEART-FAILURE RISK STRATIFICATION")
    print(f"EXPERIMENT: {args.dataset.upper()} | CONFIG: {args.config}")
    print("=" * 80)

    # 1. Inspect Environment & Verify GPU
    env_info = get_environment_info()
    print(f"Device: {env_info['device_name']} | CUDA Runtime: {env_info['cuda_version']} | VRAM: {env_info['vram_total_mb']} MB")
    device = verify_gpu_or_raise(mandatory=True)

    # 2. Load Configuration
    config = ProjectConfig.from_yaml(args.config)
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.smoke_test:
        config.training.max_epochs = 2
        config.training.batch_size = 4

    config.experiment.checkpoint_dir = args.checkpoint_dir
    config.experiment.output_dir = args.output_dir

    # 3. Locate Dataset Partitions
    proc_dir = os.path.join(config.data.processed_dir, args.dataset)
    train_npz = os.path.join(proc_dir, "train.npz")
    val_npz = os.path.join(proc_dir, "val.npz")

    if not os.path.exists(train_npz) or not os.path.exists(val_npz):
        print(f"\nCRITICAL: Processed data not found at: {proc_dir}")
        print(f"Please run data preparation first:\n  python scripts/prepare_data.py --dataset {args.dataset}")
        sys.exit(1)

    print(f"\nLoading partitions from {proc_dir}...")
    train_samples = load_npz_partition(train_npz)
    val_samples = load_npz_partition(val_npz)

    if args.smoke_test:
        train_samples = train_samples[:config.training.smoke_test_samples]
        val_samples = val_samples[:16]

    print(f"  Train cohort: {len(train_samples)} windows ({sum(s['label'] for s in train_samples):.0f} positive)")
    print(f"  Val cohort:   {len(val_samples)} windows ({sum(s['label'] for s in val_samples):.0f} positive)")

    train_dataset = MultimodalECGDataset(train_samples)
    val_dataset = MultimodalECGDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        pin_memory=(device.type == "cuda"),
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        pin_memory=(device.type == "cuda"),
        num_workers=config.training.num_workers,
    )

    # 4. Instantiate Trainer & Execute Training
    trainer = Trainer(config, checkpoint_dir=args.checkpoint_dir, output_dir=args.output_dir)
    print("\nModel Architecture Summary:")
    print(f"  ECG CNN Encoder: Conv1D Residual stages ({config.model.ecg_encoder.conv_channels})")
    print(f"  Vital TCN Encoder: Dilated Causal Conv1D (dilations {config.model.vital_encoder.dilation_schedule})")
    print(f"  Temporal Transformer: {config.model.transformer.num_layers} layers, {config.model.transformer.nhead} heads, d_model={config.model.transformer.d_model}")
    print(f"  Multimodal Fusion: Bidirectional Cross-Attention (embed_dim={config.model.cross_attention.embed_dim})")
    print(f"  Total Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"  Foundation Initialization: {trainer.model.provenance['status']}")
    print("-" * 80)

    results = trainer.fit(train_loader=train_loader, val_loader=val_loader, resume_from=args.resume)

    print("=" * 80)
    print("TRAINING RUN COMPLETE")
    print(f"Best Validation AUROC: {results['best_val_auroc']:.4f} across {results['epochs_trained']} epochs.")
    print(f"Checkpoints saved to:   {os.path.abspath(args.checkpoint_dir)}")
    print(f"Experiment logs to:     {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
