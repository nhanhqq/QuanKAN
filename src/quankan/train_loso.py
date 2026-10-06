"""Backward-compatible LOSO runner imports."""

from __future__ import annotations

from .evaluation.metrics import evaluate
from .training.cli import build_parser, main
from .training.engine import train_fold
from .training.loso import run_loso
from .training.rng import seed_everything

__all__ = [
    "build_parser",
    "evaluate",
    "main",
    "run_loso",
    "seed_everything",
    "train_fold",
]

if __name__ == "__main__":
    main()
