"""Regression checks against the author's updated LaTeX specification."""

import json
from pathlib import Path

import numpy as np
import pennylane as qml
import pytest
import torch
from torch.nn import functional as F

from quankan.losses import (
    LearnableLossWeights,
    compute_losses,
    cross_subject_prototype_loss,
)
from quankan.models.feature_enhancer import FeatureEnhancer, SpectralContextEncoder
from quankan.models.graph_encoder import LearnedSpatialPooling, TemporalConvBlock
from quankan.models.network import QuanKAN
from quankan.models.quantum import QuantumBranch
from quankan.training.cli import DEFAULT_CONFIG, load_args
from quankan.training.engine import train_fold


def arguments(*extra):
    return load_args(
        [
            "--dataset",
            "seed",
            "--data-root",
            "unused",
            "--seeds",
            "1",
            "2",
            "3",
            "4",
            "5",
            *extra,
        ]
    )


def test_packaged_defaults_match_public_config():
    root = Path(__file__).resolve().parents[1]
    assert json.loads(DEFAULT_CONFIG.read_text()) == json.loads(
        (root / "configs/base/default.json").read_text()
    )
    assert Path(arguments().locs_path).is_file()


def test_geometry_cannot_be_changed_to_legacy_v4():
    with pytest.raises(ValueError, match="paper geometry"):
        arguments("--embedding-dim", "256")
    with pytest.raises(ValueError, match="manuscript"):
        QuanKAN(torch.eye(62), 3, 14, spatial_tokens=8)


def test_duplicate_seeds_do_not_count_as_five_runs():
    with pytest.raises(ValueError, match="distinct"):
        arguments("--seeds", "1", "1", "2", "3", "4")


def test_eq4_smoothing_and_stft_have_no_extra_post_smoothing_mask():
    encoder = SpectralContextEncoder()
    assert len(encoder.head) == 2
    assert isinstance(encoder.head[0], torch.nn.Conv2d)
    assert isinstance(encoder.head[1], torch.nn.AdaptiveAvgPool2d)
    x = torch.randn(2, 16, 62, 5)
    x[0, 7:] = 0
    valid = (x.abs().sum((2, 3)) > 0).float()[:, None, :]
    signal = x.mean(2).transpose(1, 2)
    count = valid.sum(-1, keepdim=True).clamp_min(1)
    mean = (signal * valid).sum(-1, keepdim=True) / count
    variance = ((signal - mean).square() * valid).sum(-1, keepdim=True) / count
    signal = (signal - mean) / variance.sqrt().clamp_min(1e-4) * valid
    smoothed = F.avg_pool1d(signal, 3, stride=1, padding=1)
    power = (
        torch.stft(
            smoothed.reshape(10, 16),
            16,
            hop_length=4,
            window=torch.hann_window(16),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        .abs()
        .square()
    )
    log_power = torch.log1p(power).reshape(2, 5, 9, 5)
    expected = (log_power - log_power.mean((-2, -1), keepdim=True)) / log_power.std(
        (-2, -1), keepdim=True, unbiased=False
    ).clamp_min(1e-4)
    assert torch.allclose(encoder.make_spectrogram(x), expected)
    assert torch.count_nonzero(encoder.make_spectrogram(torch.zeros_like(x))) == 0


def test_eq12_13_use_raw_queries_and_original_values():
    pool = LearnedSpatialPooling()
    x = torch.randn(2, 62, 32)
    scores = torch.einsum("kh,bnh->bkn", pool.queries, pool.normalization(x)) / 32**0.5
    expected = torch.einsum("bkn,bnh->bkh", scores.softmax(-1), x)
    assert torch.allclose(pool(x), expected)


def test_eq10_padding_and_initial_residual_coefficients():
    enhancer = FeatureEnhancer(torch.eye(62)).eval()
    x = torch.randn(2, 16, 62, 5)
    x[1, 8:] = 0
    with torch.no_grad():
        enhanced, _ = enhancer(x)
    assert torch.count_nonzero(enhanced[1, 8:]) == 0
    assert torch.allclose(enhancer.temporal_scale.sigmoid(), torch.full((5,), 0.05))
    assert torch.count_nonzero(enhancer.graph_scale) == 0
    with pytest.raises(ValueError, match="nonnegative"):
        FeatureEnhancer(-torch.eye(62))


def test_temporal_block_starts_as_identity():
    block = TemporalConvBlock(256, 96, 5)
    x = torch.randn(2, 16, 256)
    assert torch.equal(block(x), x)


def test_circuit_gate_order_alternating_ring_and_observable_order():
    branch = QuantumBranch()
    angles = torch.zeros(2, 4, 4, 2)
    with qml.queuing.AnnotatedQueue() as queue:
        branch.circuit.func(angles, branch.rotation_weights)
    tape = qml.tape.QuantumScript.from_queue(queue)
    expected = (["RY", "RZ", "Rot"] * 4 + ["CNOT"] * 4) * 4
    assert [operation.name for operation in tape.operations] == expected
    for layer in range(4):
        cnot_gates = tape.operations[layer * 16 + 12 : layer * 16 + 16]
        wires = [[q, (q + 1) % 4] for q in range(4)]
        if layer % 2:
            wires = [pair[::-1] for pair in wires]
        assert [list(operation.wires) for operation in cnot_gates] == wires
    observables = [measurement.obs for measurement in tape.measurements]
    assert [observable.name for observable in observables[:8]] == ["PauliZ"] * 4 + [
        "PauliX"
    ] * 4
    assert [list(observable.wires) for observable in observables[8:]] == [
        [q, (q + 1) % 4] for q in range(4)
    ]
    assert branch.rotation_weights.numel() == 48


def test_prototypes_exclude_singletons_and_handle_empty_sets():
    z = torch.randn(3, 192, requires_grad=True)
    loss = cross_subject_prototype_loss(
        z, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), 3
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(z.grad) == 0


def test_quantum_objective_detaches_shared_encoder_and_classical_target():
    torch.set_num_threads(2)
    model = QuanKAN(torch.eye(62), 3, 14).eval()
    torch.nn.init.normal_(model.quantum_auxiliary_head.weight, std=0.1)
    x = torch.randn(2, 16, 62, 5, requires_grad=True)
    outputs = model(x)
    _, terms = compute_losses(
        outputs, torch.tensor([0, 1]), torch.tensor([0, 1]), 3, LearnableLossWeights()
    )
    terms["quantum"].backward()
    assert x.grad is None
    assert model.embedding_projection.weight.grad is None
    assert model.classical_classifier.base_weight.grad is None
    gradient = model.quantum_branch.rotation_weights.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert not outputs["gate"].requires_grad
    assert ((outputs["gate"] >= 0) & (outputs["gate"] <= 1)).all()


@pytest.mark.parametrize(
    "classes,subjects,expected",
    [
        (3, 14, 468661),
        (4, 14, 472599),
        (5, 15, 476730),
        (7, 19, 485378),
    ],
)
def test_actual_manuscript_model_parameter_counts(classes, subjects, expected):
    model = QuanKAN(torch.eye(62), classes, subjects)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert not hasattr(model, "intensity_head")
    assert not hasattr(model.encoder, "fc_emotion")


def test_real_engine_selects_source_checkpoint_and_tests_once(tmp_path, monkeypatch):
    from quankan.training import engine

    args = arguments("--epochs", "2", "--batch-size", "6")
    features = np.random.default_rng(1).normal(size=(48, 16, 62, 5)).astype(np.float32)
    labels = np.tile(np.arange(3), 16)
    subjects = [f"P{s}" for s in range(1, 5) for _ in range(12)]
    calls = []
    original = engine.evaluate

    def tracked(model, loader, device, **kwargs):
        calls.append(loader)
        return original(model, loader, device, **kwargs)

    monkeypatch.setattr(engine, "evaluate", tracked)
    result = train_fold(
        features,
        labels,
        subjects,
        "P4",
        3,
        np.eye(62, dtype=np.float32),
        5,
        tmp_path,
        args,
        torch.device("cpu"),
    )
    assert len(calls) == 3 and calls[0] is calls[1] and calls[2] is not calls[0]
    assert result["test_evaluations"] == 1
    checkpoint = torch.load(
        tmp_path / "checkpoints/P4/seed_5.pt", map_location="cpu", weights_only=True
    )
    restored = QuanKAN(torch.eye(62), 3, 3)
    restored.load_state_dict(checkpoint["model"])
    restored_weights = LearnableLossWeights()
    restored_weights.load_state_dict(checkpoint["loss_weights"])
    assert restored_weights.as_dict() == result["learned_loss_weights"]
    initial_weights = LearnableLossWeights(args.loss_weight_initialization).as_dict()
    assert any(
        abs(value - initial_weights[name]) > 1e-7
        for name, value in result["learned_loss_weights"].items()
    )
    assert result["training_settings"]["loss_weights_learnable"] is True
    assert result["training_settings"]["objective_parameters"] == 6
    assert (
        checkpoint["selected_quantum_strength"] == result["selected_quantum_strength"]
    )
    assert all(
        type(value) is str
        for value in checkpoint["training_settings"]["runtime_versions"].values()
    )
    audit = np.load(tmp_path / "checkpoints/P4/seed_5_audit.npz")
    target = set(range(36, 48))
    train, val = set(audit["train_indices"]), set(audit["validation_indices"])
    assert not (train & val or train & target or val & target)
    assert train | val == set(range(36))
    assert set(audit["test_indices"]) == target
    assert len(audit["test_predictions"]) == 12
    with pytest.raises(FileExistsError):
        train_fold(
            features,
            labels,
            subjects,
            "P4",
            3,
            np.eye(62),
            5,
            tmp_path,
            args,
            torch.device("cpu"),
        )
