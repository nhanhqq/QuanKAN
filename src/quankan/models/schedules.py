"""Paper-defined training schedules for adversarial and quantum pathways."""

from __future__ import annotations

import math


def adversarial_strength(epoch: int, max_epochs: int, maximum: float = 0.10) -> float:
    progress = epoch / max(max_epochs - 1, 1)
    return maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def quantum_warmup(epoch: int, max_epochs: int) -> float:
    progress = epoch / max(max_epochs - 1, 1)
    return min(max((progress - 0.10) / 0.20, 0.0), 1.0)
