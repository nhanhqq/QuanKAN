"""Backward-compatible public imports for QuanKAN data utilities."""

from __future__ import annotations

from .datasets import (
    SEED_IV_LABELS,
    SEED_VII_CLASSES,
    SEED_VII_LABELS,
    EEGDataset,
    EEGSubjectDataset,
    electrode_adjacency,
    load_dataset,
    load_seed,
    load_seediv,
    load_seedv,
    load_seedvii,
    standardize_from_source,
)

__all__ = [
    "SEED_IV_LABELS",
    "SEED_VII_CLASSES",
    "SEED_VII_LABELS",
    "EEGDataset",
    "EEGSubjectDataset",
    "electrode_adjacency",
    "load_dataset",
    "load_seed",
    "load_seediv",
    "load_seedv",
    "load_seedvii",
    "standardize_from_source",
]
