import torch
from torch.nn import functional as F

from quankan.data import SEED_VII_LABELS, electrode_adjacency
from quankan.losses import LearnableLossWeights, compute_losses
from quankan.model import KANLinear, QuanKAN, quantum_warmup


def test_cubic_kan_has_eight_basis_functions():
    layer = KANLinear(4, 3, grid_size=5, spline_order=3)
    values = torch.linspace(-0.9, 0.9, 12).reshape(3, 4)
    assert layer.b_splines(values).shape == (3, 4, 8)
    assert layer(values).shape == (3, 3)


def test_quantum_objective_is_mean_squared_error_and_weights_train():
    batch_size = 6
    num_classes = 3
    outputs = {
        "emotion_logits": torch.randn(batch_size, num_classes, requires_grad=True),
        "classical_logits": torch.randn(batch_size, num_classes, requires_grad=True),
        "frontend_logits": torch.randn(batch_size, num_classes, requires_grad=True),
        "subject_logits": torch.randn(batch_size, 2, requires_grad=True),
        "embedding": torch.randn(batch_size, 192, requires_grad=True),
        "quantum_auxiliary_logits": torch.randn(
            batch_size, num_classes, requires_grad=True
        ),
    }
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    subject_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    weights = LearnableLossWeights()
    total, terms = compute_losses(
        outputs, labels, subject_ids, weights, num_classes
    )
    target = F.one_hot(labels, num_classes).float() - F.softmax(
        outputs["classical_logits"].detach(), dim=-1
    )
    assert torch.allclose(
        terms["quantum"], F.mse_loss(outputs["quantum_auxiliary_logits"], target)
    )
    total.backward()
    assert all(parameter.grad is not None for parameter in weights.values.values())


def test_quantum_warmup_matches_stated_interval():
    assert quantum_warmup(0, 11) == 0.0
    assert abs(quantum_warmup(3, 11) - 1.0) < 1e-12
    assert quantum_warmup(10, 11) == 1.0


def test_dataset_graph_and_seedvii_labels():
    assert len(SEED_VII_LABELS) == 80
    adjacency = electrode_adjacency("channel_62_pos.locs")
    assert adjacency.shape == (62, 62)
    assert torch.isfinite(torch.from_numpy(adjacency)).all()


def test_paper_model_forward_shapes():
    model = QuanKAN(torch.eye(62), num_classes=3, num_subjects=14).eval()
    assert model.encoder.spatial_projection.in_features == 512
    assert model.encoder.spatial_projection.out_features == 256
    assert model.encoder.temporal_attention.in_features == 256
    assert model.embedding_projection.in_features == 256
    assert model.embedding_projection.out_features == 192
    features = torch.randn(2, 16, 62, 5)
    features[1, 12:] = 0
    with torch.no_grad():
        outputs = model(features)
    assert outputs["emotion_logits"].shape == (2, 3)
    assert outputs["embedding"].shape == (2, 192)
    assert outputs["frontend_logits"].shape == (2, 3)
