import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from .ecg_cnn import ECG1DCNNEncoder
from .vital_tcn import VitalSignsTCNEncoder
from .transformer import TemporalTransformerEncoder
from .cross_attention import BidirectionalCrossAttentionFusion
from .foundation import FoundationModelInitializer
from ..config import ModelConfig


class MultimodalHeartFailureRiskModel(nn.Module):
    """
    Proposed Architecture:
    Foundation-Guided Multimodal Temporal Learning for Short-Horizon Heart-Failure Risk Stratification.
    
    Components:
    - Branch A: ECG 1D-CNN Encoder
    - Branch B: Vital-sign TCN with missingness masking
    - Branch C: Temporal Transformer reasoning over sequences
    - Branch D: Bidirectional cross-attention multimodal fusion
    - Branch E: Pretrained foundation weight initialization interface
    - Binary classification head (logit & probability)
    
    Supported Ablation Modes:
    1. 'full': Full multimodal model (CNN + TCN + Transformer + Cross-Attention)
    2. 'ecg_only': Disables vitals, runs ECG CNN + Transformer + Classifier
    3. 'vitals_only': Disables ECG, runs Vital TCN + Transformer + Classifier
    4. 'late_fusion': Runs ECG CNN and Vital TCN independently, pools and concatenates without cross-attention
    5. 'cnn_tcn_no_transformer': Runs CNN and TCN, pools directly, skips Transformer and Cross-Attention
    6. 'transformer_no_cross_attn': Uses Transformer on concatenated tokens, but without bidirectional cross-attention
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.architecture = config.architecture

        d_model = config.transformer.d_model

        # Branch A: ECG Encoder
        self.ecg_encoder = ECG1DCNNEncoder(
            in_channels=config.ecg_encoder.in_channels,
            conv_channels=config.ecg_encoder.conv_channels,
            kernel_sizes=config.ecg_encoder.kernel_sizes,
            stride_downsample=config.ecg_encoder.stride_downsample,
            dropout=config.ecg_encoder.dropout,
            proj_dim=config.ecg_encoder.proj_dim,
        )

        # Branch B: Vital Signs TCN Encoder
        self.vital_encoder = VitalSignsTCNEncoder(
            in_channels=config.vital_encoder.in_channels,
            num_channels=config.vital_encoder.num_channels,
            kernel_size=config.vital_encoder.kernel_size,
            dilation_schedule=config.vital_encoder.dilation_schedule,
            dropout=config.vital_encoder.dropout,
            proj_dim=config.vital_encoder.proj_dim,
            use_missingness_indicator=config.vital_encoder.use_missingness_indicator,
        )

        # Branch C: Temporal Transformer
        self.ecg_transformer = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=config.transformer.nhead,
            num_layers=config.transformer.num_layers,
            dim_feedforward=config.transformer.dim_feedforward,
            dropout=config.transformer.dropout,
            max_seq_len=config.transformer.max_seq_len,
        )
        self.vital_transformer = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=config.transformer.nhead,
            num_layers=config.transformer.num_layers,
            dim_feedforward=config.transformer.dim_feedforward,
            dropout=config.transformer.dropout,
            max_seq_len=config.transformer.max_seq_len,
        )

        # Branch D: Bidirectional Cross-Attention Multimodal Fusion
        self.cross_attention = BidirectionalCrossAttentionFusion(
            embed_dim=config.cross_attention.embed_dim,
            num_heads=config.cross_attention.num_heads,
            dropout=config.cross_attention.dropout,
            fusion_gate=config.cross_attention.fusion_gate,
        )

        # Branch E: Foundation Initialization
        self.foundation_initializer = FoundationModelInitializer(
            model_name=config.foundation.foundation_model_name,
            weights_path=config.foundation.pretrained_weights_path,
            freeze_loaded_layers=config.foundation.freeze_foundation,
        )
        self.provenance = self.foundation_initializer.initialize_temporal_transformer(self.ecg_transformer)

        # Binary Classification Head
        head_in_dim = d_model
        if self.architecture in ["late_fusion", "cnn_tcn_no_transformer", "transformer_no_cross_attn"]:
            head_in_dim = d_model * 2

        self.classifier = nn.Sequential(
            nn.Linear(head_in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, config.num_classes),
        )

    def forward(
        self,
        ecg: torch.Tensor,
        vitals: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inputs:
            ecg: (Batch, C_ecg, T_ecg)
            vitals: (Batch, C_vital, T_vital)
            mask: (Batch, C_vital, T_vital) [1 = observed, 0 = missing]
        Output:
            dict containing:
                'logits': Tensor (Batch, 1)
                'probs': Tensor (Batch, 1)
                'fused_repr': Tensor (Batch, D)
        """
        # Ablation 1: ECG-only
        if self.architecture == "ecg_only":
            h_ecg = self.ecg_encoder(ecg)
            h_ecg_t = self.ecg_transformer(h_ecg)
            pooled = h_ecg_t.mean(dim=1)
            logits = self.classifier(pooled)
            probs = torch.sigmoid(logits)
            return {"logits": logits, "probs": probs, "fused_repr": pooled}

        # Ablation 2: Vitals-only
        if self.architecture == "vitals_only":
            h_vit = self.vital_encoder(vitals, mask)
            h_vit_t = self.vital_transformer(h_vit)
            pooled = h_vit_t.mean(dim=1)
            logits = self.classifier(pooled)
            probs = torch.sigmoid(logits)
            return {"logits": logits, "probs": probs, "fused_repr": pooled}

        # Encode both modalities
        h_ecg = self.ecg_encoder(ecg)             # (B, T_ecg_seq, D)
        h_vit = self.vital_encoder(vitals, mask)  # (B, T_vit_seq, D)

        # Ablation 4: CNN + TCN without Transformer
        if self.architecture == "cnn_tcn_no_transformer":
            ecg_pool = h_ecg.mean(dim=1)
            vit_pool = h_vit.mean(dim=1)
            fused = torch.cat([ecg_pool, vit_pool], dim=-1)
            logits = self.classifier(fused)
            probs = torch.sigmoid(logits)
            return {"logits": logits, "probs": probs, "fused_repr": fused}

        # Temporal Transformer Reasoning
        h_ecg_t = self.ecg_transformer(h_ecg)
        h_vit_t = self.vital_transformer(h_vit)

        # Ablation 3: Late-fusion baseline (no cross-attention)
        if self.architecture == "late_fusion":
            ecg_pool = h_ecg_t.mean(dim=1)
            vit_pool = h_vit_t.mean(dim=1)
            fused = torch.cat([ecg_pool, vit_pool], dim=-1)
            logits = self.classifier(fused)
            probs = torch.sigmoid(logits)
            return {"logits": logits, "probs": probs, "fused_repr": fused}

        # Ablation 5: Transformer without cross-attention
        if self.architecture == "transformer_no_cross_attn":
            ecg_pool = h_ecg_t.mean(dim=1)
            vit_pool = h_vit_t.mean(dim=1)
            fused = torch.cat([ecg_pool, vit_pool], dim=-1)
            logits = self.classifier(fused)
            probs = torch.sigmoid(logits)
            return {"logits": logits, "probs": probs, "fused_repr": fused}

        # Mode 6: Full proposed model with Bidirectional Cross-Attention
        fused_vec, ecg_cross, vit_cross = self.cross_attention(h_ecg=h_ecg_t, h_vital=h_vit_t)
        logits = self.classifier(fused_vec)
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probs": probs,
            "fused_repr": fused_vec,
            "ecg_cross": ecg_cross,
            "vit_cross": vit_cross,
        }
