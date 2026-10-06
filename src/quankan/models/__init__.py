"""Modular QuanKAN model components."""

from __future__ import annotations

from .adversarial import GradientReversal, grad_reverse
from .feature_enhancer import (
    FeatureEnhancer,
    MobileNetV3Block,
    SpectralContextEncoder,
    SqueezeExcitation,
)
from .graph_encoder import (
    DenseGraphAttention,
    LearnedSpatialPooling,
    SpatioTemporalGraphEncoder,
    TemporalConvBlock,
    sinusoidal_encoding,
)
from .kan import KANLinear
from .network import QuanKAN
from .quantum import QuantumBranch
from .schedules import adversarial_strength, quantum_warmup

__all__ = [
    "DenseGraphAttention",
    "FeatureEnhancer",
    "GradientReversal",
    "KANLinear",
    "LearnedSpatialPooling",
    "MobileNetV3Block",
    "QuanKAN",
    "QuantumBranch",
    "SpatioTemporalGraphEncoder",
    "SpectralContextEncoder",
    "SqueezeExcitation",
    "TemporalConvBlock",
    "adversarial_strength",
    "grad_reverse",
    "quantum_warmup",
    "sinusoidal_encoding",
]
