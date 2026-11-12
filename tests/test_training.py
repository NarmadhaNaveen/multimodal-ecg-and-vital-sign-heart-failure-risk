import pytest
import os
import torch
from src.ecg_risk.training.losses import WeightedBinaryCrossEntropyLoss, FocalBinaryLoss
from src.ecg_risk.training.checkpoint import CheckpointManager
from src.ecg_risk.config import ModelConfig
from src.ecg_risk.models.multimodal import MultimodalHeartFailureRiskModel


def test_weighted_bce_loss_and_focal_loss():
    logits = torch.tensor([[2.0], [-1.0], [0.5]], dtype=torch.float32)
    targets = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

    loss_fn = WeightedBinaryCrossEntropyLoss(pos_weight=3.0)
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    focal_fn = FocalBinaryLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_fn(logits, targets)
    assert torch.isfinite(focal_loss)
    assert focal_loss.item() > 0.0


def test_checkpoint_save_and_restoration(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(checkpoint_dir=ckpt_dir)

    cfg = ModelConfig()
    model = MultimodalHeartFailureRiskModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Save
    saved_file = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=5,
        val_metric=0.92,
        config={"test": True},
        is_best=True,
    )
    assert os.path.exists(saved_file)

    # Restore into fresh model
    new_model = MultimodalHeartFailureRiskModel(cfg)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    info = manager.load_checkpoint(saved_file, new_model, optimizer=new_opt, device="cpu")

    assert info["epoch"] == 5
    assert info["val_metric"] == 0.92

    # Verify model parameters match exactly
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
