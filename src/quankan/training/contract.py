"""Validate manuscript geometry and record choices absent from the manuscript."""

from __future__ import annotations

import argparse
import math
import platform

import numpy as np
import pennylane
import sklearn
import torch

SOURCE_AUTHORITY = "updated LaTeX supplied by the author on 2026-10-06"
SUBJECT_COUNTS = {"seed": 15, "seediv": 15, "seedv": 16, "seedvii": 20}


def validate_settings(args: argparse.Namespace) -> None:
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        raise ValueError("the manuscript requires five distinct random seeds")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("seeds must be integers between 0 and 2**32-1")
    if (args.spatial_tokens, args.embedding_dim) != (16, 192):
        raise ValueError(
            "paper geometry is fixed: spatial_tokens=16, embedding_dim=192"
        )
    if args.sessions != [1, 2, 3]:
        raise ValueError(
            "SEED/SEED-IV require all sessions [1,2,3]; VII loads all four"
        )
    if args.epochs < 1 or args.epochs > 300 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("require 1<=epochs<=300, batch_size>=1 and workers>=0")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    for name in ("channel_drop", "time_mask", "band_drop", "augmentation_probability"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    for name in ("noise_std", "gain_jitter", "weight_decay", "minimum_lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("backbone_lr", "head_lr", "frontend_lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.quantum_device != "default.qubit":
        raise ValueError(
            "this release uses the validated default.qubit state-vector backend"
        )


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "pennylane": pennylane.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
