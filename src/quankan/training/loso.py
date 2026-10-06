"""Dataset-level five-seed strict leave-one-subject-out orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..datasets import electrode_adjacency, load_dataset
from .contract import SUBJECT_COUNTS, validate_settings
from .engine import _subject_number, train_fold
from .rng import seed_everything


def run_loso(args: argparse.Namespace) -> dict:
    validate_settings(args)
    features, labels, subjects, num_classes = load_dataset(
        args.dataset, args.data_root, tuple(args.sessions)
    )
    adjacency = electrode_adjacency(args.locs_path)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(args.output_dir) / args.dataset
    if output_dir.exists() and (
        any(output_dir.rglob("seed_*.pt"))
        or (output_dir / "fold_results.json").exists()
        or (output_dir / "summary.json").exists()
    ):
        raise FileExistsError(
            f"use a new output directory; completed artifacts exist in {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    target_subjects = sorted(set(subjects), key=_subject_number)
    if len(target_subjects) != SUBJECT_COUNTS[args.dataset]:
        raise ValueError(
            f"{args.dataset} requires {SUBJECT_COUNTS[args.dataset]} subjects; "
            f"loaded {len(target_subjects)}"
        )
    for target_subject in target_subjects:
        for seed in args.seeds:
            seed_everything(seed)
            result = train_fold(
                features=features,
                labels=labels,
                subjects=subjects,
                target_subject=target_subject,
                num_classes=num_classes,
                adjacency=adjacency,
                seed=seed,
                output_dir=output_dir,
                args=args,
                device=device,
            )
            fold_results.append(result)
            print(json.dumps(result, sort_keys=True))
            with (output_dir / "fold_results.json").open("w", encoding="utf-8") as file:
                json.dump(fold_results, file, indent=2)

    subject_results = []
    for target_subject in target_subjects:
        seed_results = [
            result
            for result in fold_results
            if result["target_subject"] == target_subject
        ]
        subject_results.append(
            {
                "target_subject": target_subject,
                "accuracy": float(
                    np.mean([result["accuracy"] for result in seed_results])
                ),
                "macro_f1": float(
                    np.mean([result["macro_f1"] for result in seed_results])
                ),
            }
        )
    summary = {
        "protocol": "leave-one-subject-out; source-subject validation; held-out test once",
        "dataset": args.dataset,
        "subjects": subject_results,
        "accuracy_mean": float(np.mean([item["accuracy"] for item in subject_results])),
        "accuracy_sd": float(np.std([item["accuracy"] for item in subject_results])),
        "macro_f1_mean": float(np.mean([item["macro_f1"] for item in subject_results])),
        "macro_f1_sd": float(np.std([item["macro_f1"] for item in subject_results])),
        "seeds": args.seeds,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary
