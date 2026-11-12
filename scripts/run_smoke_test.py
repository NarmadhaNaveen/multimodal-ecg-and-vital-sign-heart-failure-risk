import sys
import os
import torch
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ecg_risk.config import ModelConfig, TrainingConfig
from src.ecg_risk.models.multimodal import MultimodalHeartFailureRiskModel
from src.ecg_risk.training.losses import WeightedBinaryCrossEntropyLoss
from src.ecg_risk.training.checkpoint import CheckpointManager
from src.ecg_risk.evaluation.metrics import evaluate_predictions

def main():
    print("=" * 80)
    print("GPU ACCELERATED SMOKE TEST: MULTIMODAL ECG HEART-FAILURE RISK PIPELINE")
    print("=" * 80)

    # 1. CUDA Availability Check
    print("[1/8] Verifying CUDA Availability...")
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available. Smoke test aborted.")
        sys.exit(1)
    device = torch.device("cuda:0")
    print(f"PASS: CUDA active. GPU: {torch.cuda.get_device_name(0)}")

    # 2. Model Parameters on GPU
    print("[2/8] Instantiating Multimodal Model & transferring to GPU...")
    model_cfg = ModelConfig()
    model = MultimodalHeartFailureRiskModel(model_cfg).to(device)
    model.train()
    for name, p in model.named_parameters():
        assert p.is_cuda, f"Parameter {name} is not on CUDA!"
    print(f"PASS: All model parameters ({sum(p.numel() for p in model.parameters()):,} parameters) on GPU.")

    # 3. Batch Tensors on GPU
    print("[3/8] Generating synthetic paired observation batch on GPU...")
    batch_size = 4
    t_ecg = int(600 * 250)      # 10 min context at 250 Hz = 150,000 samples
    # For quick smoke test, test temporal slice
    t_ecg_smoke = 15000
    t_vital_smoke = 600

    ecg_batch = torch.randn(batch_size, 2, t_ecg_smoke, device=device)
    vital_batch = torch.randn(batch_size, 6, t_vital_smoke, device=device)
    mask_batch = torch.ones(batch_size, 6, t_vital_smoke, device=device)
    target_batch = torch.tensor([[1.0], [0.0], [1.0], [0.0]], device=device)
    print(f"PASS: Batches allocated on CUDA: ECG={tuple(ecg_batch.shape)}, Vitals={tuple(vital_batch.shape)}")

    # 4. Forward Pass
    print("[4/8] Executing forward pass through CNN + TCN + Transformer + Cross-Attention...")
    with torch.amp.autocast("cuda"):
        out = model(ecg=ecg_batch, vitals=vital_batch, mask=mask_batch)
    logits = out["logits"]
    probs = out["probs"]
    assert logits.shape == (batch_size, 1), f"Unexpected logits shape: {logits.shape}"
    assert probs.shape == (batch_size, 1), f"Unexpected probs shape: {probs.shape}"
    print(f"PASS: Forward pass succeeded. Output logits shape: {tuple(logits.shape)}")

    # 5. Finite Loss Calculation
    print("[5/8] Calculating weighted binary cross-entropy loss...")
    criterion = WeightedBinaryCrossEntropyLoss(pos_weight=2.0).to(device)
    loss = criterion(logits, target_batch)
    assert torch.isfinite(loss).item(), "Loss is NaN or infinite!"
    print(f"PASS: Loss calculated successfully: {loss.item():.4f} (finite).")

    # 6. Backpropagation & Optimizer Step
    print("[6/8] Executing backpropagation and AdamW optimizer step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    print("PASS: Gradient computation and weight updates completed successfully.")

    # 7. Checkpoint Save and Reload
    print("[7/8] Testing checkpoint serialization and restoration...")
    ckpt_mgr = CheckpointManager("checkpoints/smoke_test")
    saved_path = ckpt_mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        val_metric=0.95,
        config={"smoke_test": True},
        is_best=True,
    )
    assert os.path.exists(saved_path), "Checkpoint file was not written!"
    loaded_info = ckpt_mgr.load_checkpoint(saved_path, model, optimizer=optimizer, device="cuda")
    assert loaded_info["epoch"] == 1, "Failed to restore epoch!"
    print("PASS: Checkpoint save and reload verified.")

    # 8. Metric Calculation
    print("[8/8] Testing evaluation metric calculation...")
    y_true = np.array([1, 0, 1, 0])
    y_pred_probs = probs.detach().cpu().numpy().squeeze(-1)
    metrics = evaluate_predictions(y_true, y_pred_probs, threshold=0.5)
    assert "auroc" in metrics and "accuracy" in metrics and "ece" in metrics
    print(f"PASS: Metrics computed: AUROC={metrics['auroc']:.3f}, Acc={metrics['accuracy']:.3f}, ECE={metrics['ece']:.3f}")

    print("=" * 80)
    print("ALL 8 GPU SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
