import math
from typing import Optional

import pennylane as qml
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise
        step = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1)
        knots = knots * step + grid_range[0]
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = torch.rand(
                self.grid_size + 1, self.in_features, self.out_features,
                device=self.grid.device, dtype=self.grid.dtype,
            )
            noise = (noise - 0.5) * self.scale_noise / self.grid_size
            interior_grid = self.grid.T[self.spline_order:-self.spline_order]
            coefficients = self._curve_to_coefficients(interior_grid, noise)
            self.spline_weight.copy_(self.scale_spline * coefficients)
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x: Tensor) -> Tensor:
        knots = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= knots[:, :-1]) & (x < knots[:, 1:])).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            left = (x - knots[:, :-(order + 1)]) / (
                knots[:, order:-1] - knots[:, :-(order + 1)] + 1e-8
            )
            right = (knots[:, order + 1:] - x) / (
                knots[:, order + 1:] - knots[:, 1:-order] + 1e-8
            )
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def _curve_to_coefficients(self, x: Tensor, y: Tensor) -> Tensor:
        basis = self.b_splines(x).transpose(0, 1)
        targets = y.transpose(0, 1)
        solution = torch.linalg.lstsq(basis, targets).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flattened = x.reshape(-1, self.in_features)
        base = F.linear(F.silu(flattened), self.scale_base * self.base_weight)
        spline_weights = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = F.linear(
            self.b_splines(flattened).reshape(flattened.size(0), -1),
            spline_weights.reshape(self.out_features, -1),
        )
        return (base + spline).reshape(*original_shape[:-1], self.out_features)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


def grad_reverse(x: Tensor, strength: float) -> Tensor:
    return GradientReversal.apply(x, strength)


def adversarial_strength(epoch: int, max_epochs: int, maximum: float = 0.10) -> float:
    progress = epoch / max(max_epochs - 1, 1)
    return maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def quantum_warmup(epoch: int, max_epochs: int) -> float:
    progress = epoch / max(max_epochs - 1, 1)
    return min(max((progress - 0.10) / 0.20, 0.0), 1.0)


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
    def __init__(self, in_channels: int, expanded: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.Hardswish(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded, expanded, kernel_size=3, stride=stride, padding=1,
                groups=expanded, bias=False,
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
            nn.BatchNorm2d(context_dim),
            nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def make_spectrogram(self, x: Tensor) -> Tensor:
        valid = (x.abs().sum(dim=(2, 3)) > 0).to(x.dtype)
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        signal = x.mean(dim=2).transpose(1, 2)
        mask = valid[:, None, :]
        mean = (signal * mask).sum(dim=-1, keepdim=True) / denominator[:, None, :]
        variance = (
            (signal - mean).square() * mask
        ).sum(dim=-1, keepdim=True) / denominator[:, None, :]
        signal = ((signal - mean) / variance.sqrt().clamp_min(1e-4)) * mask
        signal = F.avg_pool1d(
            signal, kernel_size=3, stride=1, padding=1, count_include_pad=True
        ) * mask
        batch_size, bands, steps = signal.shape
        spectrum = torch.stft(
            signal.reshape(batch_size * bands, steps),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(signal),
            center=True,
            pad_mode="constant",
            return_complex=True,
        ).abs().square()
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
    def __init__(self, adjacency: Tensor, num_nodes: int = 62, bands: int = 5,
                 num_classes: int = 4) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.bands = bands
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(f"adjacency must have shape {(num_nodes, num_nodes)}")
        adjacency = adjacency.clamp_min(0.0)
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
        node_gain = gains[:, :self.num_nodes, None]
        band_gain = gains[:, self.num_nodes:][:, None, None, :]
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


class DenseGraphAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, heads: int = 8,
                 dropout: float = 0.3) -> None:
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


def sinusoidal_encoding(steps: int, channels: int, device: torch.device,
                        dtype: torch.dtype) -> Tensor:
    positions = torch.arange(steps, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, channels, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / channels)
    )
    encoding = torch.zeros(steps, channels, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[:encoding[:, 1::2].shape[1]])
    return encoding


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, bottleneck: int, kernel_size: int,
                 dropout: float = 0.10) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck),
            nn.GELU(),
            nn.Conv1d(
                bottleneck, bottleneck, kernel_size=kernel_size,
                padding=kernel_size // 2, groups=bottleneck, bias=False,
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
    def __init__(self, num_nodes: int = 62, bands: int = 5,
                 graph_dim: int = 32, spatial_tokens: int = 16) -> None:
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


class QuantumBranch(nn.Module):
    def __init__(self, embedding_dim: int = 192,
                 device_name: str = "default.qubit") -> None:
        super().__init__()
        self.num_qubits = 4
        self.num_layers = 4
        self.compression = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.angle_map = KANLinear(32, 4 * 4 * 2)
        self.device = qml.device(device_name, wires=self.num_qubits)
        self.rotation_weights = nn.Parameter(0.05 * torch.randn(4, 4, 3))

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs: Tensor, weights: Tensor):
            for layer in range(4):
                for qubit in range(4):
                    qml.RY(inputs[:, layer, qubit, 0], wires=qubit)
                    qml.RZ(inputs[:, layer, qubit, 1], wires=qubit)
                for qubit in range(4):
                    qml.Rot(*weights[layer, qubit], wires=qubit)
                if layer % 2 == 0:
                    for qubit in range(4):
                        qml.CNOT(wires=[qubit, (qubit + 1) % 4])
                else:
                    for qubit in range(4):
                        qml.CNOT(wires=[(qubit + 1) % 4, qubit])
            measurements = [qml.expval(qml.PauliZ(q)) for q in range(4)]
            measurements.extend(qml.expval(qml.PauliX(q)) for q in range(4))
            measurements.extend(
                qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % 4))
                for q in range(4)
            )
            return measurements

        self.circuit = circuit
        self.measurement_map = KANLinear(12, 32)

    def forward(self, x: Tensor) -> Tensor:
        angles = math.pi * torch.tanh(self.angle_map(self.compression(x)))
        angles = angles.reshape(-1, 4, 4, 2)
        measurements = self.circuit(angles.cpu(), self.rotation_weights.cpu())
        measurements = torch.stack(measurements, dim=1).to(device=x.device, dtype=x.dtype)
        return self.measurement_map(measurements)


class QuanKAN(nn.Module):
    def __init__(self, adjacency: Tensor, num_classes: int, num_subjects: int,
                 num_nodes: int = 62, bands: int = 5,
                 quantum_device: str = "default.qubit",
                 spatial_tokens: int = 16, embedding_dim: int = 192) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.frontend = FeatureEnhancer(adjacency, num_nodes, bands, num_classes)
        self.encoder = SpatioTemporalGraphEncoder(
            num_nodes, bands, spatial_tokens=spatial_tokens
        )
        self.embedding_dim = embedding_dim
        self.embedding_normalization = nn.LayerNorm(256)
        self.embedding_projection = nn.Linear(256, embedding_dim)
        self.classical_dropout = nn.Dropout(0.20)
        self.classical_classifier = KANLinear(embedding_dim, num_classes)
        self.quantum_branch = QuantumBranch(embedding_dim, quantum_device)
        self.quantum_projection = nn.Linear(32, embedding_dim)
        nn.init.normal_(self.quantum_projection.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.quantum_projection.bias)
        self.quantum_scale = nn.Parameter(torch.zeros(()))
        self.quantum_auxiliary_head = nn.Linear(32, num_classes)
        nn.init.zeros_(self.quantum_auxiliary_head.weight)
        nn.init.zeros_(self.quantum_auxiliary_head.bias)
        self.hybrid_classifier = KANLinear(embedding_dim, num_classes)
        self.subject_classifier = nn.Linear(embedding_dim, num_subjects)

    def forward(self, x: Tensor, adversarial_coefficient: float = 0.0,
                quantum_strength: float = 1.0) -> dict[str, Tensor]:
        enhanced, frontend_logits = self.frontend(x)
        temporal = self.encoder(enhanced)
        embedding = F.gelu(
            self.embedding_projection(self.embedding_normalization(temporal))
        )
        classical_feature = self.classical_dropout(embedding)
        classical_logits = self.classical_classifier(classical_feature)
        quantum_feature = self.quantum_branch(embedding.detach())
        probabilities = F.softmax(classical_logits.detach(), dim=-1)
        normalized_entropy = -(
            probabilities * torch.log(probabilities + 1e-8)
        ).sum(dim=-1, keepdim=True) / math.log(self.num_classes)
        gate = normalized_entropy.clamp(0.0, 1.0).pow(1.5)
        alpha = 0.25 * torch.sigmoid(self.quantum_scale) * quantum_strength
        hybrid_feature = (
            classical_feature + gate * alpha * self.quantum_projection(quantum_feature)
        )
        emotion_logits = self.hybrid_classifier(hybrid_feature)
        quantum_auxiliary_logits = self.quantum_auxiliary_head(quantum_feature)
        subject_logits = self.subject_classifier(
            grad_reverse(embedding, adversarial_coefficient)
        )
        return {
            "emotion_logits": emotion_logits,
            "classical_logits": classical_logits,
            "frontend_logits": frontend_logits,
            "subject_logits": subject_logits,
            "embedding": embedding,
            "quantum_auxiliary_logits": quantum_auxiliary_logits,
            "quantum_feature": quantum_feature,
            "gate": gate,
            "normalized_entropy": normalized_entropy,
        }
