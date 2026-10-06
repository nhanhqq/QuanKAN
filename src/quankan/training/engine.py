"""One-fold training with source-only validation and a single final test pass."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..augmentation import RobustEEGAugment
from ..datasets import EEGDataset, EEGSubjectDataset, standardize_from_source
from ..evaluation.metrics import evaluate
from ..losses import LearnableLossWeights, compute_losses
from ..models.network import QuanKAN
from ..models.schedules import adversarial_strength, quantum_warmup
from .contract import SOURCE_AUTHORITY, SUBJECT_COUNTS, runtime_versions
from .rng import seed_everything


def _subject_number(subject: str) -> int:
    return int(subject[1:])


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
    seed_everything(seed)
    checkpoint_dir = output_dir / "checkpoints" / target_subject
    checkpoint_path = checkpoint_dir / f"seed_{seed}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing fold: {checkpoint_path}"
        )
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
        features, train_indices
    )
    source_subjects = sorted(
        {subjects[index] for index in source_indices}, key=_subject_number
    )
    if {subjects[index] for index in train_indices} != set(source_subjects):
        raise ValueError("source split must retain every source subject in training")
    subject_to_index = {subject: index for index, subject in enumerate(source_subjects)}
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
        apply_probability=args.augmentation_probability,
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
        spatial_tokens=args.spatial_tokens,
        embedding_dim=args.embedding_dim,
    ).to(device)
    initialization = args.loss_weight_initialization
    loss_weights = LearnableLossWeights(initialization).to(device)
    training_settings = {
        "source_authority": SOURCE_AUTHORITY,
        "loss_weight_provenance": "author correction: learnable; initial values and clamp parameterization restored from original release",
        "loss_weight_parameterization": "lambda = max(raw_lambda, 0)",
        "loss_weights_learnable": True,
        "objective_parameters": sum(
            parameter.numel() for parameter in loss_weights.parameters()
        ),
        "implementation_choices": {
            "source_validation": "stratified 80/20 trial split by default",
            "normalization": "train-fold-only population statistics per band",
            "augmentation_application": "V4 probability 0.85 by default",
            "gat": "V4 dense attention, 8 heads and dropout 0.30",
            "selected_schedule": "validation and test use the selected epoch's quantum warm-up",
        },
        "runtime_versions": runtime_versions(),
        "source_subjects": source_subjects,
        "num_emotion_classes": num_classes,
        "subject_to_index": subject_to_index,
        "validation_scheme": "stratified source-pool trial holdout",
        "normalization_fit": "source training trials only",
        "quantum_execution": "CPU state-vector; classical model on requested device",
        "stored_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_fraction": args.validation_fraction,
        "validation_selection_metric": "accuracy",
        "workers": args.workers,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "frontend_lr": args.frontend_lr,
        "weight_decay": args.weight_decay,
        "minimum_lr": args.minimum_lr,
        "noise_std": args.noise_std,
        "gain_jitter": args.gain_jitter,
        "channel_drop": args.channel_drop,
        "time_mask": args.time_mask,
        "band_drop": args.band_drop,
        "augmentation_probability": args.augmentation_probability,
        "spatial_tokens": args.spatial_tokens,
        "shared_embedding_dim": args.embedding_dim,
        "quantum_device": args.quantum_device,
        "loss_weight_initialization": initialization.as_dict(),
        "loss_weight_lr": args.head_lr,
    }
    paper_settings = {
        "epochs": 300,
        "batch_size": 32,
        "backbone_lr": 3e-4,
        "head_lr": 5e-4,
        "frontend_lr": 8e-4,
        "weight_decay": 1e-4,
        "minimum_lr": 1e-6,
        "noise_std": 0.035,
        "gain_jitter": 0.10,
        "channel_drop": 0.05,
        "time_mask": 0.10,
        "band_drop": 0.15,
    }
    deviations = {
        name: {"manuscript": value, "actual": training_settings[name]}
        for name, value in paper_settings.items()
        if training_settings[name] != value
    }
    if len(source_subjects) != SUBJECT_COUNTS[args.dataset] - 1:
        deviations["source_subject_count"] = {"actual": len(source_subjects)}
    training_settings["execution_kind"] = (
        "development" if deviations else "paper_settings"
    )
    training_settings["deviations_from_paper_settings"] = deviations
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
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
            {"params": model.frontend.parameters(), "lr": args.frontend_lr},
            {"params": quantum_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_lr
    )

    best_accuracy = -1.0
    best_epoch = 0
    best_model_state = None
    best_weight_state = None
    best_quantum_strength = 0.0
    epoch_history = []
    for epoch in range(args.epochs):
        model.train()
        loss_weights.train()
        quantum_strength = quantum_warmup(epoch, args.epochs)
        batch_losses = []
        for batch, targets, batch_subject_ids in train_loader:
            batch = batch.to(device)
            targets = targets.to(device)
            batch_subject_ids = batch_subject_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch,
                adversarial_coefficient=adversarial_strength(epoch, args.epochs),
                quantum_strength=quantum_strength,
            )
            loss, _ = compute_losses(
                outputs, targets, batch_subject_ids, num_classes, loss_weights
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation_metrics = evaluate(
            model, validation_loader, device, quantum_strength=quantum_strength
        )
        epoch_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(batch_losses)),
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "quantum_strength": quantum_strength,
                "loss_weights": loss_weights.as_dict(),
            }
        )
        print(
            json.dumps(
                {"target_subject": target_subject, "seed": seed, **epoch_history[-1]}
            ),
            flush=True,
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_weight_state = copy.deepcopy(loss_weights.state_dict())
            best_quantum_strength = quantum_strength

    if best_model_state is None or best_weight_state is None:
        raise RuntimeError("source-subject validation did not select a checkpoint")
    model.load_state_dict(best_model_state)
    loss_weights.load_state_dict(best_weight_state)
    test_metrics = evaluate(
        model, test_loader, device, quantum_strength=best_quantum_strength
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint_dir / f"seed_{seed}_audit.npz",
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=target_indices,
        test_labels=np.asarray(test_metrics.pop("labels")),
        test_predictions=np.asarray(test_metrics.pop("predictions")),
    )
    result = {
        "dataset": args.dataset,
        "target_subject": target_subject,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_accuracy": best_accuracy,
        "learned_loss_weights": loss_weights.as_dict(),
        "test_evaluations": 1,
        "selected_quantum_strength": best_quantum_strength,
        "split_sizes": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(target_indices),
        },
        **test_metrics,
        "training_settings": training_settings,
    }
    torch.save(
        {
            "model": best_model_state,
            "loss_weights": best_weight_state,
            "training_settings": training_settings,
            "epoch": best_epoch,
            "seed": seed,
            "target_subject": target_subject,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "result": result,
            "epoch_history": epoch_history,
            "selected_quantum_strength": best_quantum_strength,
        },
        checkpoint_path,
    )
    return result
