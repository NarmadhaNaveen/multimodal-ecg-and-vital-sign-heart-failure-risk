import torch
import torch.nn as nn
from typing import List, Optional


class ChausalConv1d(nn.Module):
    """Causal dilated 1D convolution with left-padding to prevent future information leakage."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future padding on the right
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual TCN block with dilated causal convolutions, Weight/LayerNorm, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.res = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return out + res


class VitalSignsTCNEncoder(nn.Module):
    """
    Branch B: Vital Signs Temporal Convolutional Network.
    Accepts vitals of shape (Batch, num_vitals, T_vitals) and missingness mask of shape (Batch, num_vitals, T_vitals).
    Concatenates measurements and mask indicators so missingness is explicitly modeled.
    Applies dilated causal convolutions across time.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dilation_schedule: List[int] = None,
        dropout: float = 0.2,
        proj_dim: int = 128,
        use_missingness_indicator: bool = True,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64, 128]
        if dilation_schedule is None:
            dilation_schedule = [1, 2, 4, 8]

        self.use_mask = use_missingness_indicator
        effective_in_channels = in_channels * 2 if self.use_mask else in_channels

        layers = []
        curr_in = effective_in_channels
        for c in num_channels:
            for d in dilation_schedule:
                layers.append(TemporalBlock(curr_in, c, kernel_size=kernel_size, dilation=d, dropout=dropout))
                curr_in = c
        self.network = nn.Sequential(*layers)

        self.proj = nn.Sequential(
            nn.Conv1d(curr_in, proj_dim, kernel_size=1),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input:
            x: (Batch, in_channels, T_vital)
            mask: (Batch, in_channels, T_vital) [1 = observed, 0 = missing]
        Output:
            (Batch, seq_len, proj_dim)
        """
        if self.use_mask:
            if mask is None:
                mask = torch.ones_like(x)
            # Concatenate raw values and explicit observation indicators along channel dim
            x = torch.cat([x, mask], dim=1)

        h = self.network(x)
        h = self.proj(h)  # (B, proj_dim, T_vital)
        h = h.transpose(1, 2)  # (B, T_vital, proj_dim)
        return h
