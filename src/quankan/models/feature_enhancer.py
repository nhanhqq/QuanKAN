"""V4 spectrum-conditioned feature enhancement, aligned to paper Eqs. (2)-(10)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.reduce(scale), inplace=True)
        scale = F.hardsigmoid(self.expand(scale), inplace=True)
        return x * scale


class MobileNetV3Block(nn.Module):
    def __init__(
        self, in_channels: int, expanded: int, out_channels: int, stride: int
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.Hardswish(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded,
                expanded,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=expanded,
                bias=False,
            ),
            nn.BatchNorm2d(expanded),
            nn.Hardswish(inplace=True),
            SqueezeExcitation(expanded),
        )
        self.project = nn.Sequential(
            nn.Conv2d(expanded, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        result = self.project(self.depthwise(self.expand(x)))
        return x + result if self.use_residual else result


class SpectralContextEncoder(nn.Module):
    def __init__(self, bands: int = 5, context_dim: int = 64) -> None:
        super().__init__()
        self.n_fft = 16
        self.hop_length = 4
        self.register_buffer("window", torch.hann_window(16), persistent=False)
        self.features = nn.Sequential(
            nn.Conv2d(bands, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            MobileNetV3Block(16, 48, 24, 1),
            MobileNetV3Block(24, 72, 32, 2),
            MobileNetV3Block(32, 96, 48, 1),
            MobileNetV3Block(48, 144, 64, 2),
            MobileNetV3Block(64, 192, 64, 1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, context_dim, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )

    def make_spectrogram(self, x: Tensor) -> Tensor:
        valid = (x.abs().sum(dim=(2, 3)) > 0).to(x.dtype)
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        signal = x.mean(dim=2).transpose(1, 2)
        mask = valid[:, None, :]
        mean = (signal * mask).sum(dim=-1, keepdim=True) / denominator[:, None, :]
        variance = ((signal - mean).square() * mask).sum(
            dim=-1, keepdim=True
        ) / denominator[:, None, :]
        signal = ((signal - mean) / variance.sqrt().clamp_min(1e-4)) * mask
        signal = F.avg_pool1d(
            signal, kernel_size=3, stride=1, padding=1, count_include_pad=True
        )
        batch_size, bands, steps = signal.shape
        spectrum = (
            torch.stft(
                signal.reshape(batch_size * bands, steps),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window.to(signal),
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            .abs()
            .square()
        )
        spectrum = torch.log1p(spectrum).reshape(
            batch_size, bands, spectrum.size(-2), spectrum.size(-1)
        )
        mean = spectrum.mean(dim=(-2, -1), keepdim=True)
        std = spectrum.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
        return (spectrum - mean) / std

    def forward(self, x: Tensor) -> Tensor:
        spectrum = self.make_spectrogram(x)
        return self.head(self.features(spectrum)).flatten(1)


class FeatureEnhancer(nn.Module):
    def __init__(
        self,
        adjacency: Tensor,
        num_nodes: int = 62,
        bands: int = 5,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.bands = bands
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(f"adjacency must have shape {(num_nodes, num_nodes)}")
        if not torch.isfinite(adjacency).all() or (adjacency < 0).any():
            raise ValueError("adjacency must be finite and nonnegative")
        row_sum = adjacency.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("adjacency", adjacency / row_sum)
        self.spectral_context = SpectralContextEncoder(bands=bands, context_dim=64)
        self.conditioner = nn.Linear(64, num_nodes + bands)
        nn.init.normal_(self.conditioner.weight, std=0.005)
        nn.init.zeros_(self.conditioner.bias)
        self.context_to_input = nn.Linear(64, num_nodes * bands)
        nn.init.normal_(self.context_to_input.weight, std=0.002)
        nn.init.zeros_(self.context_to_input.bias)
        self.aux_classifier = nn.Linear(64, num_classes)
        self.temporal_filter = nn.Sequential(
            nn.Conv1d(bands, bands, kernel_size=5, padding=2, groups=bands, bias=False),
            nn.Conv1d(bands, bands, kernel_size=1, bias=False),
        )
        initial_scale = math.log(0.05 / 0.95)
        self.temporal_scale = nn.Parameter(torch.full((bands,), initial_scale))
        self.graph_scale = nn.Parameter(torch.zeros(bands))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 4 or x.shape[2:] != (self.num_nodes, self.bands):
            raise ValueError(
                f"expected [B,T,{self.num_nodes},{self.bands}], got {tuple(x.shape)}"
            )
        valid = (x.abs().sum(dim=(2, 3), keepdim=True) > 0).to(x.dtype)
        context = self.spectral_context(x)
        frontend_logits = self.aux_classifier(context)
        gains = 0.25 * torch.tanh(self.conditioner(context))
        node_gain = gains[:, : self.num_nodes, None]
        band_gain = gains[:, self.num_nodes :][:, None, None, :]
        conditioned = x * (1.0 + node_gain[:, None, :, :])
        conditioned = conditioned * (1.0 + band_gain)
        shift = 0.20 * torch.tanh(self.context_to_input(context))
        shift = shift.reshape(-1, 1, self.num_nodes, self.bands)
        conditioned = conditioned + shift * valid

        batch_size, steps, nodes, bands = conditioned.shape
        temporal = conditioned.permute(0, 2, 3, 1).reshape(
            batch_size * nodes, bands, steps
        )
        temporal = self.temporal_filter(temporal)
        temporal = temporal.reshape(batch_size, nodes, bands, steps).permute(0, 3, 1, 2)
        temporal_weight = torch.sigmoid(self.temporal_scale)[None, None, None, :]
        smoothed = torch.einsum("nm,btmf->btnf", self.adjacency, conditioned)
        graph_delta = smoothed - conditioned
        graph_weight = (0.20 * torch.tanh(self.graph_scale))[None, None, None, :]
        enhanced = conditioned + temporal_weight * temporal + graph_weight * graph_delta
        return enhanced * valid, frontend_logits
