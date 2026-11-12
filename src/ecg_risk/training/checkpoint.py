import os
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving, loading, and restoration of model and training state.
    Preserves model weights, optimizer state, scheduler state, epoch, best metric,
    and experiment configuration for exact research reproducibility.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        val_metric: float,
        config: Dict[str, Any],
        is_best: bool = False,
        filename: str = "latest.pt",
    ) -> str:
        """Saves a training checkpoint."""
        save_path = os.path.join(self.checkpoint_dir, filename)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "val_metric": val_metric,
            "config": config,
        }
        torch.save(ckpt, save_path)
        logger.info(f"Saved checkpoint to {save_path} (epoch={epoch}, val_metric={val_metric:.4f})")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pt")
            torch.save(ckpt, best_path)
            logger.info(f"Updated best model checkpoint: {best_path}")

        return save_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Loads weights and state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded model state dict from {checkpoint_path}")

        if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("Restored optimizer state.")

        if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            logger.info("Restored learning rate scheduler state.")

        return {
            "epoch": ckpt.get("epoch", 0),
            "val_metric": ckpt.get("val_metric", 0.0),
            "config": ckpt.get("config", {}),
        }
