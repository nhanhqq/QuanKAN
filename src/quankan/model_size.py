"""Count stored parameters of the manuscript architecture, without a forward pass."""

from __future__ import annotations

import json

import torch

from .losses import LearnableLossWeights
from .models.network import QuanKAN
from .training.contract import SOURCE_AUTHORITY, SUBJECT_COUNTS, runtime_versions


def main() -> None:
    classes = {"seed": 3, "seediv": 4, "seedv": 5, "seedvii": 7}
    rows = []
    objective_parameters = sum(p.numel() for p in LearnableLossWeights().parameters())
    for dataset, subjects in SUBJECT_COUNTS.items():
        model = QuanKAN(torch.eye(62), classes[dataset], subjects - 1)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        rows.append(
            {
                "dataset": dataset,
                "source_subject_classes": subjects - 1,
                "stored_parameters": parameters,
                "training_objective_parameters": objective_parameters,
                "total_trainable_parameters": parameters + objective_parameters,
                "fp32_weights_mib": parameters * 4 / 1024**2,
            }
        )
    print(
        json.dumps(
            {
                "source_authority": SOURCE_AUTHORITY,
                "runtime_versions": runtime_versions(),
                "models": rows,
                "note": "Table XII in the manuscript describes different stored models; "
                "MACs/FLOPs have not been reprofiled here.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
