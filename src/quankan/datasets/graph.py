"""Fixed electrode-affinity graph construction from channel coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def electrode_adjacency(locs_path: str) -> np.ndarray:
    rows = pd.read_csv(
        locs_path,
        sep=r"\s+",
        header=None,
        names=["id", "theta_deg", "radius", "label"],
    )
    theta = np.deg2rad(rows["theta_deg"].to_numpy())
    radius = rows["radius"].to_numpy()
    coordinates = np.stack(
        (
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.sqrt(np.clip(1 - radius**2, 0, 1)),
        ),
        axis=1,
    )
    if coordinates.shape != (62, 3):
        raise ValueError(f"expected 62 electrode coordinates, got {coordinates.shape}")
    coordinates = coordinates * 20.0
    adjacency = np.zeros((62, 62), dtype=np.float32)
    for row in range(62):
        for column in range(62):
            if row == column:
                adjacency[row, column] = 1.0
            else:
                distance = np.linalg.norm(coordinates[row] - coordinates[column])
                adjacency[row, column] = min(1.0, 5.0 / (distance**2 + 1e-8))
    symmetric_pairs = [
        (1, 2),
        (4, 5),
        (6, 14),
        (7, 13),
        (8, 12),
        (9, 11),
        (15, 23),
        (16, 22),
        (17, 21),
        (18, 20),
        (24, 32),
        (25, 31),
        (26, 30),
        (27, 29),
        (33, 41),
        (42, 50),
        (43, 49),
        (44, 48),
        (45, 47),
        (51, 57),
        (52, 56),
        (53, 55),
        (59, 61),
        (58, 62),
    ]
    for left, right in symmetric_pairs:
        adjacency[left - 1, right - 1] = 1.0
        adjacency[right - 1, left - 1] = 1.0
    return adjacency
