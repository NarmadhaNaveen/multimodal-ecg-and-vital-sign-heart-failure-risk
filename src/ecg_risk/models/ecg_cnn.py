import torch
import torch.nn as nn
from typing import List, Optional


class Conv1DBlock(nn.Module):
    """Residual 1D Convolutional Block with batch/layer norm and GELU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()

        # Residual shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class ECG1DCNNEncoder(nn.Module):
    """
    Branch A: Multi-channel ECG 1D-CNN Encoder.
    Accepts raw ECG waveforms of shape (Batch, in_channels, time_samples).
    Downsamples the high-frequency waveform while extracting multiscale morphological features,
    producing a projected temporal sequence of shape (Batch, seq_len, proj_dim) for temporal reasoning.
    """

    def __init__(
        self,
        in_channels: int = 2,
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        stride_downsample: List[int] = None,
        dropout: float = 0.2,
        proj_dim: int = 128,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [7, 7, 5, 3]
        if stride_downsample is None:
            stride_downsample = [4, 4, 4, 4]

        assert len(conv_channels) == len(kernel_sizes) == len(stride_downsample)

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels[0], kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(conv_channels[0]),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual hierarchical stages
        layers = []
        curr_in = conv_channels[0]
        for out_c, k, s in zip(conv_channels, kernel_sizes, stride_downsample):
            layers.append(Conv1DBlock(curr_in, out_c, kernel_size=k, stride=s, dropout=dropout))
            curr_in = out_c
        self.stages = nn.Sequential(*layers)

        # Projection to common embedding space for temporal reasoning
        self.proj = nn.Sequential(
            nn.Conv1d(curr_in, proj_dim, kernel_size=1),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, in_channels, T_ecg)
        Output: (Batch, seq_len, proj_dim)
        """
        h = self.stem(x)
        h = self.stages(h)
        h = self.proj(h)  # (B, proj_dim, seq_len)
        # Transpose to (Batch, seq_len, proj_dim) for sequence / transformer models
        h = h.transpose(1, 2)
        return h
