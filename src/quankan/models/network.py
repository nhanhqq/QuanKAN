"""Paper-aligned QuanKAN network composition."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adversarial import grad_reverse
from .feature_enhancer import FeatureEnhancer
from .graph_encoder import SpatioTemporalGraphEncoder
from .kan import KANLinear
from .quantum import QuantumBranch


class QuanKAN(nn.Module):
    def __init__(
        self,
        adjacency: Tensor,
        num_classes: int,
        num_subjects: int,
        num_nodes: int = 62,
        bands: int = 5,
        quantum_device: str = "default.qubit",
        spatial_tokens: int = 16,
        embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if (num_nodes, bands, spatial_tokens, embedding_dim) != (62, 5, 16, 192):
            raise ValueError(
                "the manuscript requires 62 electrodes, 5 bands, "
                "16 spatial tokens and a 192-dimensional shared latent"
            )
        if num_subjects < 1:
            raise ValueError("num_subjects must count the source subjects in this fold")
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

    def forward(
        self,
        x: Tensor,
        adversarial_coefficient: float = 0.0,
        quantum_strength: float = 1.0,
    ) -> dict[str, Tensor]:
        enhanced, frontend_logits = self.frontend(x)
        temporal = self.encoder(enhanced)
        embedding = F.gelu(
            self.embedding_projection(self.embedding_normalization(temporal))
        )
        classical_feature = self.classical_dropout(embedding)
        classical_logits = self.classical_classifier(classical_feature)
        quantum_feature = self.quantum_branch(embedding.detach())
        probabilities = F.softmax(classical_logits.detach(), dim=-1)
        normalized_entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(
            dim=-1, keepdim=True
        ) / math.log(self.num_classes)
        gate = normalized_entropy.clamp(0.0, 1.0).pow(1.5)
        alpha = 0.25 * torch.sigmoid(self.quantum_scale) * quantum_strength
        hybrid_feature = classical_feature + gate * alpha * self.quantum_projection(
            quantum_feature
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
