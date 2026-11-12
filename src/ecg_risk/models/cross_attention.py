import torch
import torch.nn as nn
from typing import Tuple, Optional


class CrossAttentionBlock(nn.Module):
    """
    Directional Cross-Attention Block where Query attends to Key/Value from another modality.
    Applies MultiheadAttention with Pre-LayerNorm and MLP residual feedforward.
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        query: (B, T_q, D)
        key_value: (B, T_kv, D)
        returns: (B, T_q, D)
        """
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)
        attn_out, _ = self.mha(query=q_norm, key=kv_norm, value=kv_norm)
        h = query + self.drop(attn_out)
        out = h + self.mlp(self.norm_ff(h))
        return out


class BidirectionalCrossAttentionFusion(nn.Module):
    """
    Branch D: Bidirectional Cross-Attention Multimodal Fusion.
    - ECG attends to Vital signs (learns vital context for ECG abnormalities)
    - Vital signs attend to ECG (learns electrophysiological correlates for hemodynamic swings)
    - Gated or projected fusion combines pooled multimodal contexts into a unified representation.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_gate: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_gate = fusion_gate

        # Bidirectional cross-attention modules
        self.ecg_attends_vitals = CrossAttentionBlock(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.vitals_attend_ecg = CrossAttentionBlock(d_model=embed_dim, nhead=num_heads, dropout=dropout)

        if fusion_gate:
            # Dynamic gating network determining relative modality weighting
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 2),
                nn.Softmax(dim=-1),
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
        else:
            self.fusion_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )

    def forward(
        self,
        h_ecg: torch.Tensor,    # (Batch, T_ecg, D)
        h_vital: torch.Tensor,  # (Batch, T_vital, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_vector: (Batch, embed_dim)
            ecg_cross: (Batch, T_ecg, embed_dim)
            vital_cross: (Batch, T_vital, embed_dim)
        """
        # Cross-attend
        ecg_cross = self.ecg_attends_vitals(query=h_ecg, key_value=h_vital)
        vital_cross = self.vitals_attend_ecg(query=h_vital, key_value=h_ecg)

        # Global temporal pooling (mean across time)
        ecg_pool = ecg_cross.mean(dim=1)      # (Batch, D)
        vital_pool = vital_cross.mean(dim=1)  # (Batch, D)

        if self.fusion_gate:
            combined = torch.cat([ecg_pool, vital_pool], dim=-1)  # (B, 2*D)
            weights = self.gate(combined)  # (B, 2)
            w_ecg = weights[:, 0:1]
            w_vit = weights[:, 1:2]
            fused_raw = w_ecg * ecg_pool + w_vit * vital_pool
            fused_vector = self.fusion_proj(fused_raw)
        else:
            combined = torch.cat([ecg_pool, vital_pool], dim=-1)
            fused_vector = self.fusion_proj(combined)

        return fused_vector, ecg_cross, vital_cross
