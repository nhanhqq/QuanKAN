import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .augmentation import RobustEEGAugment
from .data import (
    EEGDataset,
    EEGSubjectDataset,
    electrode_adjacency,
    load_dataset,
    standardize_from_source,
)
from .losses import LearnableLossWeights, compute_losses
from .model import QuanKAN, adversarial_strength, quantum_warmup


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: QuanKAN, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features.to(device))
            labels.extend(targets.tolist())
            predictions.extend(outputs["emotion_logits"].argmax(dim=1).cpu().tolist())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "samples": len(labels),
    }


def _subject_number(subject: str) -> int:
    return int(subject.removeprefix("P"))


def train_fold(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: list[str],
    target_subject: str,
    num_classes: int,
    adjacency: np.ndarray,
    seed: int,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    target_indices = np.asarray(
        [index for index, subject in enumerate(subjects) if subject == target_subject]
    )
    source_indices = np.asarray(
        [index for index, subject in enumerate(subjects) if subject != target_subject]
    )
    if target_indices.size == 0 or source_indices.size == 0:
        raise ValueError(f"invalid LOSO fold for {target_subject}")
    train_indices, validation_indices = train_test_split(
        source_indices,
        test_size=args.validation_fraction,
        stratify=labels[source_indices],
        random_state=seed,
    )
    standardized, feature_mean, feature_std = standardize_from_source(
        features, source_indices
    )
    source_subjects = sorted(
        {subjects[index] for index in source_indices}, key=_subject_number
    )
    subject_to_index = {
        subject: index for index, subject in enumerate(source_subjects)
    }
    train_subject_ids = np.asarray(
        [subject_to_index[subjects[index]] for index in train_indices],
        dtype=np.int64,
    )
    augmenter = RobustEEGAugment(
        noise_std=args.noise_std,
        gain=args.gain_jitter,
        channel_drop=args.channel_drop,
        time_mask=args.time_mask,
        band_drop_probability=args.band_drop,
    )
    train_loader = DataLoader(
        EEGSubjectDataset(
            standardized[train_indices],
            labels[train_indices],
            train_subject_ids,
            transform=augmenter,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        EEGDataset(standardized[validation_indices], labels[validation_indices]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        EEGDataset(standardized[target_indices], labels[target_indices]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = QuanKAN(
        adjacency=torch.as_tensor(adjacency),
        num_classes=num_classes,
        num_subjects=len(source_subjects),
        quantum_device=args.quantum_device,
    ).to(device)
    loss_weights = LearnableLossWeights().to(device)
    backbone_parameters = (
        list(model.encoder.parameters())
        + list(model.embedding_normalization.parameters())
        + list(model.embedding_projection.parameters())
    )
    head_parameters = (
        list(model.classical_classifier.parameters())
        + list(model.hybrid_classifier.parameters())
        + list(model.subject_classifier.parameters())
        + list(loss_weights.parameters())
    )
    quantum_parameters = (
        list(model.quantum_branch.parameters())
        + list(model.quantum_projection.parameters())
        + [model.quantum_scale]
        + list(model.quantum_auxiliary_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 3e-4},
            {"params": head_parameters, "lr": 5e-4},
            {"params": model.frontend.parameters(), "lr": 8e-4},
            {"params": quantum_parameters, "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_accuracy = -1.0
    best_epoch = 0
    best_model_state = None
    best_weight_state = None
    for epoch in range(args.epochs):
        model.train()
        loss_weights.train()
        for batch, targets, batch_subject_ids in train_loader:
            batch = batch.to(device)
            targets = targets.to(device)
            batch_subject_ids = batch_subject_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch,
                adversarial_coefficient=adversarial_strength(epoch, args.epochs),
                quantum_strength=quantum_warmup(epoch, args.epochs),
            )
            loss, _ = compute_losses(
                outputs, targets, batch_subject_ids, loss_weights, num_classes
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
        validation_metrics = evaluate(model, validation_loader, device)
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_weight_state = copy.deepcopy(loss_weights.state_dict())

    if best_model_state is None or best_weight_state is None:
        raise RuntimeError("source-subject validation did not select a checkpoint")
    model.load_state_dict(best_model_state)
    loss_weights.load_state_dict(best_weight_state)
    test_metrics = evaluate(model, test_loader, device)
    result = {
        "dataset": args.dataset,
        "target_subject": target_subject,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        **test_metrics,
        "learned_loss_weights": {
            name: float(value.detach().cpu())
            for name, value in loss_weights().items()
        },
    }
    checkpoint_dir = output_dir / "checkpoints" / target_subject
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_model_state,
            "loss_weights": best_weight_state,
            "epoch": best_epoch,
            "seed": seed,
            "target_subject": target_subject,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "result": result,
        },
        checkpoint_dir / f"seed_{seed}.pt",
    )
    return result


def run_loso(args: argparse.Namespace) -> dict:
    if len(args.seeds) != 5:
        raise ValueError("the paper reports five random seeds; pass exactly five values")
    features, labels, subjects, num_classes = load_dataset(
        args.dataset, args.data_root, tuple(args.sessions)
    )
    adjacency = electrode_adjacency(args.locs_path)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    target_subjects = sorted(set(subjects), key=_subject_number)
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
            result for result in fold_results
            if result["target_subject"] == target_subject
        ]
        subject_results.append({
            "target_subject": target_subject,
            "accuracy": float(np.mean([result["accuracy"] for result in seed_results])),
            "macro_f1": float(np.mean([result["macro_f1"] for result in seed_results])),
        })
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("seed", "seediv", "seedv", "seedvii"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--locs-path", default="channel_62_pos.locs")
    parser.add_argument("--sessions", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--quantum-device", default="default.qubit")
    parser.add_argument("--noise-std", type=float, default=0.035)
    parser.add_argument("--gain-jitter", type=float, default=0.10)
    parser.add_argument("--channel-drop", type=float, default=0.05)
    parser.add_argument("--time-mask", type=float, default=0.10)
    parser.add_argument("--band-drop", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_loso(args), indent=2))


if __name__ == "__main__":
    main()
