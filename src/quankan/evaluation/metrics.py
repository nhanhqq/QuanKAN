"""Metrics computed from a completed validation or held-out evaluation pass."""

from __future__ import annotations

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from ..models.network import QuanKAN


def evaluate(
    model: QuanKAN,
    loader: DataLoader,
    device: torch.device,
    quantum_strength: float = 1.0,
) -> dict:
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features.to(device), quantum_strength=quantum_strength)
            labels.extend(targets.tolist())
            predictions.extend(outputs["emotion_logits"].argmax(dim=1).cpu().tolist())
    if not labels:
        raise ValueError("cannot evaluate an empty validation/test loader")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(range(model.num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "samples": len(labels),
        "labels": labels,
        "predictions": predictions,
    }
