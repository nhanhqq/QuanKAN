"""Graph attention, learned electrode pooling, and temporal aggregation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DenseGraphAttention(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, heads: int = 8, dropout: float = 0.3
    ) -> None:
        super().__init__()
        if out_features % heads:
            raise ValueError("out_features must be divisible by heads")
        self.heads = heads
        self.out_features = out_features
        self.qkv = nn.Linear(in_features, 3 * out_features, bias=False)
        self.projection = nn.Linear(out_features, out_features, bias=False)
        self.normalization = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, nodes, _ = x.shape
        head_dim = self.out_features // self.heads
        qkv = self.qkv(x).reshape(batch, nodes, 3, self.heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        score = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
        attention = self.dropout(F.softmax(score, dim=-1))
        result = attention @ value
        result = result.transpose(1, 2).reshape(batch, nodes, self.out_features)
        result = self.projection(result)
        if x.shape[-1] == self.out_features:
            result = result + x
        return self.normalization(result)


class LearnedSpatialPooling(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_tokens: int = 16) -> None:
        super().__init__()
        self.queries = nn.Parameter(0.02 * torch.randn(num_tokens, hidden_dim))
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        keys = self.normalization(x)
        scores = torch.einsum("kh,bnh->bkn", self.queries, keys)
        weights = F.softmax(scores / math.sqrt(x.shape[-1]), dim=-1)
        return torch.einsum("bkn,bnh->bkh", weights, x)


def sinusoidal_encoding(
    steps: int, channels: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    positions = torch.arange(steps, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, channels, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / channels)
    )
    encoding = torch.zeros(steps, channels, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class TemporalConvBlock(nn.Module):
    def __init__(
        self, channels: int, bottleneck: int, kernel_size: int, dropout: float = 0.10
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck),
            nn.GELU(),
            nn.Conv1d(
                bottleneck,
                bottleneck,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=bottleneck,
                bias=False,
            ),
            nn.BatchNorm1d(bottleneck),
            nn.GELU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.block[-2].weight)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.block(x.transpose(1, 2)).transpose(1, 2)
        return x + residual


class SpatioTemporalGraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int = 62,
        bands: int = 5,
        graph_dim: int = 32,
        spatial_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.bands = bands
        self.graph_attention_1 = DenseGraphAttention(bands, graph_dim, heads=8)
        self.graph_attention_2 = DenseGraphAttention(graph_dim, graph_dim, heads=8)
        self.spatial_pooling = LearnedSpatialPooling(graph_dim, spatial_tokens)
        spatial_dim = spatial_tokens * graph_dim
        self.spatial_normalization = nn.LayerNorm(spatial_dim)
        self.spatial_projection = nn.Linear(spatial_dim, 256)
        self.temporal_conv_3 = TemporalConvBlock(256, 96, 3)
        self.temporal_conv_5 = TemporalConvBlock(256, 96, 5)
        self.temporal_attention = nn.Linear(256, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch, steps, nodes, bands = x.shape
        if nodes != self.num_nodes or bands != self.bands:
            raise ValueError(
                f"expected node/band dimensions {(self.num_nodes, self.bands)}, "
                f"got {(nodes, bands)}"
            )
        electrodes = x.reshape(batch * steps, nodes, bands)
        electrodes = self.graph_attention_1(electrodes)
        electrodes = self.graph_attention_2(electrodes)
        tokens = self.spatial_pooling(electrodes)
        spatial = tokens.reshape(batch, steps, -1)
        temporal = self.spatial_projection(self.spatial_normalization(spatial))
        positions = sinusoidal_encoding(steps, 256, x.device, x.dtype)
        temporal = temporal + positions.unsqueeze(0)
        temporal = self.temporal_conv_3(temporal)
        temporal = self.temporal_conv_5(temporal)
        weights = F.softmax(self.temporal_attention(temporal).squeeze(-1), dim=1)
        return torch.sum(temporal * weights.unsqueeze(-1), dim=1)
