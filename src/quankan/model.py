"""Backward-compatible public imports for the modular QuanKAN model."""

from __future__ import annotations

from .models import (
    DenseGraphAttention,
    FeatureEnhancer,
    GradientReversal,
    KANLinear,
    LearnedSpatialPooling,
    MobileNetV3Block,
    QuanKAN,
    QuantumBranch,
    SpatioTemporalGraphEncoder,
    SpectralContextEncoder,
    SqueezeExcitation,
    TemporalConvBlock,
    adversarial_strength,
    grad_reverse,
    quantum_warmup,
    sinusoidal_encoding,
)

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
