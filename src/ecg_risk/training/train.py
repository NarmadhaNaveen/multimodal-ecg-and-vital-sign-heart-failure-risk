import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List

from ..config import ProjectConfig
from ..device import verify_gpu_or_raise, log_gpu_memory
from ..models.multimodal import MultimodalHeartFailureRiskModel
from .losses import WeightedBinaryCrossEntropyLoss
from .checkpoint import CheckpointManager
from .experiment import ExperimentTracker
from ..evaluation.metrics import evaluate_predictions

logger = logging.getLogger(__name__)


class Trainer:
    """
    Research-grade GPU-accelerated multimodal training pipeline.
    Implements mixed precision (AMP), gradient accumulation, gradient clipping,
    early stopping, per-epoch GPU profiling, and metric evaluation.
    """

    def __init__(
        self,
        config: ProjectConfig,
        model: Optional[MultimodalHeartFailureRiskModel] = None,
        checkpoint_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir or config.experiment.checkpoint_dir
        self.output_dir = output_dir or config.experiment.output_dir

        # Reproducibility seed
        torch.manual_seed(config.experiment.seed)
        np.random.seed(config.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.experiment.seed)

        # Device selection & verification
        req_gpu = config.experiment.device.lower() == "cuda"
        self.device = verify_gpu_or_raise(mandatory=req_gpu)

        # Model instantiation
        self.model = model or MultimodalHeartFailureRiskModel(config.model)
        self.model.to(self.device)

        # Loss function with positive class weighting
        self.criterion = WeightedBinaryCrossEntropyLoss(pos_weight=config.training.pos_weight).to(self.device)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.max_epochs,
            eta_min=1e-6,
        )

        # Mixed precision setup
        self.use_amp = config.training.mixed_precision and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Management utilities
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_dir)
        self.tracker = ExperimentTracker(self.output_dir, config.experiment.name)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Runs a single training epoch with gradient accumulation and AMP."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        accum_steps = max(1, self.config.training.gradient_accumulation_steps)

        self.optimizer.zero_grad()
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            # Batch tensors to GPU
            ecg = batch["ecg"].to(self.device, non_blocking=True)
            vitals = batch["vitals"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(ecg=ecg, vitals=vitals, mask=mask)
                loss = self.criterion(outputs["logits"], targets)
                loss = loss / accum_steps

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Encountered NaN or Inf loss at epoch {epoch}, step {step}.")

            self.scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(1, num_batches)
        mem = log_gpu_memory(f"Epoch {epoch} ")

        return {
            "train_loss": avg_loss,
            "train_epoch_time_sec": epoch_time,
            "gpu_allocated_mb": mem["allocated_mb"],
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluates model on validation or test cohort."""
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_targets = []
        all_logits = []

        for batch in dataloader:
            ecg = batch["ecg"].to(self.device, non_blocking=True)
            vitals = batch["vitals"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(ecg=ecg, vitals=vitals, mask=mask)
                loss = self.criterion(outputs["logits"], targets)

            total_loss += loss.item()
            all_logits.append(outputs["logits"].detach().cpu().numpy())
            all_probs.append(outputs["probs"].detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

        logits = np.concatenate(all_logits, axis=0).squeeze(-1)
        probs = np.concatenate(all_probs, axis=0).squeeze(-1)
        targets = np.concatenate(all_targets, axis=0).squeeze(-1)

        metrics = evaluate_predictions(y_true=targets, y_probs=probs, logits=logits)
        metrics["loss"] = total_loss / max(1, len(dataloader))
        return metrics, probs, targets

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes full training loop with early stopping, checkpointing, and scheduler step.
        """
        start_epoch = 1
        best_val_auroc = 0.0
        patience_counter = 0

        if resume_from is not None and os.path.exists(resume_from):
            ckpt_info = self.checkpoint_mgr.load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler, device=self.device.type
            )
            start_epoch = ckpt_info["epoch"] + 1
            best_val_auroc = ckpt_info.get("val_metric", 0.0)
            logger.info(f"Resuming training from epoch {start_epoch} (previous best AUROC: {best_val_auroc:.4f})")

        logger.info(f"Starting training on {self.device.type.upper()} for max {self.config.training.max_epochs} epochs...")

        for epoch in range(start_epoch, self.config.training.max_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics, _, _ = self.evaluate(val_loader)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            val_auroc = val_metrics.get("auroc", 0.0)
            val_f1 = val_metrics.get("f1_score", 0.0)

            logger.info(
                f"[Epoch {epoch:02d}/{self.config.training.max_epochs}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | "
                f"Val AUPRC: {val_metrics.get('auprc', 0.0):.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            epoch_log = {
                "epoch": epoch,
                **train_metrics,
                "val_loss": val_metrics["loss"],
                "val_auroc": val_auroc,
                "val_auprc": val_metrics.get("auprc", 0.0),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
                "val_f1": val_f1,
                "val_ece": val_metrics.get("ece", 0.0),
                "lr": current_lr,
            }
            self.tracker.log_epoch(epoch, epoch_log)

            is_best = val_auroc > best_val_auroc
            if is_best:
                best_val_auroc = val_auroc
                patience_counter = 0
                self.checkpoint_mgr.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, best_val_auroc,
                    config=vars(self.config), is_best=True
                )
            else:
                patience_counter += 1

            # Save latest checkpoint
            self.checkpoint_mgr.save_checkpoint(
                self.model, self.optimizer, self.scheduler, epoch, val_auroc,
                config=vars(self.config), is_best=False, filename="latest.pt"
            )

            if patience_counter >= self.config.training.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no AUROC improvement for {patience_counter} epochs).")
                break

        # Restore best model weights
        best_path = os.path.join(self.checkpoint_dir, "best.pt")
        if os.path.exists(best_path):
            self.checkpoint_mgr.load_checkpoint(best_path, self.model, device=self.device.type)
            logger.info("Restored best model weights for final evaluation.")

        self.tracker.finalize({"best_val_auroc": best_val_auroc, "epochs_trained": epoch})
        return {"best_val_auroc": best_val_auroc, "epochs_trained": epoch}
