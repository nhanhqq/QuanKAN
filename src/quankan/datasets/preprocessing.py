"""Leakage-safe source-training-fold feature standardization."""

from __future__ import annotations

import numpy as np


def standardize_from_source(
    features: np.ndarray, source_indices: np.ndarray
) -> tuple[np.ndarray, list[float], list[float]]:
    if features.ndim != 4 or not np.isfinite(features).all():
        raise ValueError("features must be finite [trials,time,electrodes,bands]")
    if source_indices.size == 0:
        raise ValueError("source training indices cannot be empty")
    valid = np.any(np.abs(features) > 0, axis=(2, 3))
    source_valid = valid[source_indices]
    means, standard_deviations = [], []
    for band in range(features.shape[-1]):
        values = features[source_indices, :, :, band][source_valid]
        if values.size == 0:
            raise ValueError("source training trials contain no valid windows")
        means.append(float(values.mean()))
        standard_deviations.append(float(max(values.std(), 1e-4)))
    standardized = features.copy()
    for band, (mean, standard_deviation) in enumerate(zip(means, standard_deviations)):
        values = (standardized[:, :, :, band] - mean) / standard_deviation
        standardized[:, :, :, band] = np.where(valid[:, :, None], values, 0.0)
    return standardized, means, standard_deviations
