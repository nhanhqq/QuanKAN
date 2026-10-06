"""Exercise the real fold engine on synthetic data; produces no paper results."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .training.cli import load_args
from .training.engine import train_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    output = options.output_dir or Path(tempfile.mkdtemp(prefix="quankan-smoke-"))
    args = load_args(
        [
            "--dataset",
            "seed",
            "--data-root",
            "synthetic",
            "--seeds",
            "1",
            "2",
            "3",
            "4",
            "5",
            "--epochs",
            "2",
            "--batch-size",
            "6",
            "--device",
            "cpu",
        ]
    )
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    features = rng.normal(size=(48, 16, 62, 5)).astype(np.float32)
    features[::2, 12:] = 0
    subjects = [f"P{subject}" for subject in range(1, 5) for _ in range(12)]
    labels = np.tile(np.arange(3), 16)
    result = train_fold(
        features,
        labels,
        subjects,
        "P4",
        3,
        np.eye(62, dtype=np.float32),
        5,
        output,
        args,
        torch.device("cpu"),
    )
    report = {
        "purpose": "synthetic execution check; not scientific performance evidence",
        "output_dir": str(output),
        "result": result,
    }
    (output / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
