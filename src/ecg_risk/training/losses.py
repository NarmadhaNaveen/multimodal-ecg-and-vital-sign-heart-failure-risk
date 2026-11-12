import torch
import torch.nn as nn
from typing import Optional


class WeightedBinaryCrossEntropyLoss(nn.Module):
    """
    Numerically stable Binary Cross-Entropy with Logits
    incorporating positive class weighting to counteract severe imbalance in acute clinical outcomes.
    """

    def __init__(self, pos_weight: float = 2.5):
        super().__init__()
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight", weight_tensor)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (Batch, 1)
        targets: (Batch, 1) or (Batch,)
        """
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        return self.loss_fn(logits, targets)


class FocalBinaryLoss(nn.Module):
    """
    Focal Loss for focusing training on hard boundary / borderline clinical examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce
        return loss.mean()
