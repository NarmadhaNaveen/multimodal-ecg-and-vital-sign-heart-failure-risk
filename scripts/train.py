
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
    """Load a preprocessed .npz partition into sample dictionaries."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Partition file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        required = {"ecg", "vitals", "mask", "labels", "patient_ids"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"{npz_path} is missing required arrays: {sorted(missing)}"
            )

        ecg = data["ecg"]
        vitals = data["vitals"]
        mask = data["mask"]
        labels = data["labels"]
        patient_ids = data["patient_ids"]

    n = len(labels)
    if n == 0:
        raise ValueError(f"Partition is empty: {npz_path}")

    if not all(len(arr) == n for arr in (ecg, vitals, mask, patient_ids)):
        raise ValueError(f"Array sample counts do not match in {npz_path}")

    if ecg.ndim != 3 or ecg.shape[1] != 2:
        raise ValueError(f"Expected ECG shape (N, 2, T), got {ecg.shape}")
    if vitals.ndim != 3 or vitals.shape[1] != 6:
        raise ValueError(f"Expected vitals shape (N, 6, T), got {vitals.shape}")
    if mask.shape != vitals.shape:
        raise ValueError("Vital mask shape does not match vital data shape")
    if not np.isfinite(ecg).all():
        raise ValueError(f"Non-finite ECG values found in {npz_path}")
    if not np.isfinite(vitals).all() or not np.isfinite(mask).all():
        raise ValueError(f"Non-finite vital or mask values found in {npz_path}")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Labels must be binary (0/1) in {npz_path}")

    return [
        {
            "ecg": ecg[i].astype(np.float32, copy=False),
            "vitals": vitals[i].astype(np.float32, copy=False),
            "mask": mask[i].astype(np.float32, copy=False),
            "label": float(labels[i]),
            "patient_id": str(patient_ids[i]),
            "window_idx": i,
        }
        for i in range(n)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Train Multimodal ECG & Vital-Sign Heart-Failure Risk Model."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to experiment config YAML."
    )
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "mimic", "vitaldb"],
        help="Dataset partition to train on."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override processed dataset root, e.g. data/norm_v2."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run rapid smoke-test training (1-2 epochs)."
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save logs and metrics.")
    args = parser.parse_args()

    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr must be greater than zero.")

    print("=" * 80)
    print("MULTIMODAL ECG & VITAL-SIGN HEART-FAILURE RISK STRATIFICATION")
    print(f"EXPERIMENT: {args.dataset.upper()} | CONFIG: {args.config}")
    print("=" * 80)

    # 1. Inspect environment and verify GPU.
    env_info = get_environment_info()
    print(
        f"Device: {env_info['device_name']} | "
        f"CUDA Runtime: {env_info['cuda_version']} | "
        f"VRAM: {env_info['vram_total_mb']} MB"
    )
    device = verify_gpu_or_raise(mandatory=True)

    # 2. Load configuration using the project's existing API.
    config = ProjectConfig.from_yaml(args.config)

    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr

    if args.smoke_test:
        config.training.max_epochs = 2
        config.training.batch_size = min(config.training.batch_size, 4)

    # 3. Locate dataset partitions.
    data_root = (
        args.data_dir
        if args.data_dir is not None
        else config.data.processed_dir
    )
    proc_dir = os.path.join(data_root, args.dataset)
    train_npz = os.path.join(proc_dir, "train.npz")
    val_npz = os.path.join(proc_dir, "val.npz")

    if not os.path.exists(train_npz) or not os.path.exists(val_npz):
        print(f"\nCRITICAL: Processed data not found at: {proc_dir}")
        print("Expected files: train.npz and val.npz")
        print(
            "Prepare data first, or provide the correct --data-dir.\n"
            "Example: --data-dir data/norm_v2"
        )
        sys.exit(1)

    print(f"\nLoading partitions from {proc_dir}...")
    train_samples = load_npz_partition(train_npz)
    val_samples = load_npz_partition(val_npz)

    if args.smoke_test:
        train_samples = train_samples[:config.training.smoke_test_samples]
        val_samples = val_samples[:16]

    if not train_samples or not val_samples:
        raise ValueError("Training and validation partitions must not be empty.")

    print(
        f"  Train cohort: {len(train_samples)} windows "
        f"({sum(s['label'] for s in train_samples):.0f} positive)"
    )
    print(
        f"  Val cohort:   {len(val_samples)} windows "
        f"({sum(s['label'] for s in val_samples):.0f} positive)"
    )

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

    # 4. Preserve the original Trainer API.
    trainer = Trainer(
        config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )

    print("\nModel Architecture Summary:")
    print(
        f"  ECG CNN Encoder: Conv1D Residual stages "
        f"({config.model.ecg_encoder.conv_channels})"
    )
    print(
        f"  Vital TCN Encoder: Dilated Causal Conv1D "
        f"(dilations {config.model.vital_encoder.dilation_schedule})"
    )
    print(
        f"  Temporal Transformer: {config.model.transformer.num_layers} layers, "
        f"{config.model.transformer.nhead} heads, "
        f"d_model={config.model.transformer.d_model}"
    )
    print(
        f"  Multimodal Fusion: Bidirectional Cross-Attention "
        f"(embed_dim={config.model.cross_attention.embed_dim})"
    )
    print(f"  Total Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"  Foundation Initialization: {trainer.model.provenance['status']}")
    print("-" * 80)

    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )

    print("=" * 80)
    print("TRAINING RUN COMPLETE")
    print(
        f"Best Validation AUROC: {results['best_val_auroc']:.4f} "
        f"across {results['epochs_trained']} epochs."
    )
    print(f"Checkpoints saved to:   {os.path.abspath(args.checkpoint_dir)}")
    print(f"Experiment logs to:     {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
