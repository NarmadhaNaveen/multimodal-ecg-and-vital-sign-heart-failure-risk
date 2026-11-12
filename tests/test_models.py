import pytest
import torch
from src.ecg_risk.config import ModelConfig, ECGEncoderConfig, VitalEncoderConfig, TransformerConfig, CrossAttentionConfig
from src.ecg_risk.models.ecg_cnn import ECG1DCNNEncoder
from src.ecg_risk.models.vital_tcn import VitalSignsTCNEncoder
from src.ecg_risk.models.transformer import TemporalTransformerEncoder
from src.ecg_risk.models.cross_attention import BidirectionalCrossAttentionFusion
from src.ecg_risk.models.multimodal import MultimodalHeartFailureRiskModel


def test_ecg_cnn_forward_dimensions():
    model = ECG1DCNNEncoder(in_channels=2, proj_dim=128)
    batch_size = 2
    ecg_samples = 15000  # 1 minute at 250 Hz
    x = torch.randn(batch_size, 2, ecg_samples)

    out = model(x)
    assert out.ndim == 3
    assert out.size(0) == batch_size
    assert out.size(2) == 128
    assert out.size(1) > 0


def test_vital_tcn_forward_dimensions():
    model = VitalSignsTCNEncoder(in_channels=6, proj_dim=128, use_missingness_indicator=True)
    batch_size = 2
    vital_steps = 600  # 10 minutes at 1 Hz
    x = torch.randn(batch_size, 6, vital_steps)
    mask = torch.ones(batch_size, 6, vital_steps)

    out = model(x, mask)
    assert out.shape == (batch_size, vital_steps, 128)


def test_temporal_transformer_forward():
    transformer = TemporalTransformerEncoder(d_model=128, nhead=4, num_layers=2)
    x = torch.randn(2, 50, 128)
    out = transformer(x)
    assert out.shape == (2, 50, 128)


def test_bidirectional_cross_attention_gated_fusion():
    fusion = BidirectionalCrossAttentionFusion(embed_dim=128, num_heads=4, fusion_gate=True)
    h_ecg = torch.randn(2, 60, 128)
    h_vit = torch.randn(2, 40, 128)

    fused, ecg_cross, vit_cross = fusion(h_ecg, h_vit)
    assert fused.shape == (2, 128)
    assert ecg_cross.shape == (2, 60, 128)
    assert vit_cross.shape == (2, 40, 128)


@pytest.mark.parametrize("architecture", [
    "full",
    "ecg_only",
    "vitals_only",
    "late_fusion",
    "cnn_tcn_no_transformer",
    "transformer_no_cross_attn",
])
def test_multimodal_ablations_forward_pass(architecture):
    cfg = ModelConfig(architecture=architecture)
    model = MultimodalHeartFailureRiskModel(cfg)

    ecg = torch.randn(2, 2, 15000)
    vitals = torch.randn(2, 6, 600)
    mask = torch.ones(2, 6, 600)

    out = model(ecg=ecg, vitals=vitals, mask=mask)
    assert "logits" in out
    assert "probs" in out
    assert out["logits"].shape == (2, 1)
    assert out["probs"].shape == (2, 1)
    assert torch.all((out["probs"] >= 0.0) & (out["probs"] <= 1.0))
