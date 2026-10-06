"""Paper objectives with six trainable nonnegative auxiliary-loss weights."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LossCoefficients:
    """Initial values for trainable weights; defaults match the original release."""

    classical: float = 0.20
    frontend: float = 0.25
    subject: float = 0.20
    supcon: float = 0.08
    prototype: float = 0.05
    quantum: float = 0.15

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"loss coefficient {name!r} must be finite and nonnegative"
                )

    def as_dict(self) -> dict[str, float]:
        return {
            "classical": self.classical,
            "frontend": self.frontend,
            "subject": self.subject,
            "supcon": self.supcon,
            "prototype": self.prototype,
            "quantum": self.quantum,
        }


class LearnableLossWeights(nn.Module):
    """Restore the original release's lambda=max(raw_lambda,0) parameterization."""

    def __init__(self, initialization: LossCoefficients | None = None) -> None:
        super().__init__()
        initialization = initialization or LossCoefficients()
        self.values = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(value, dtype=torch.float32))
                for name, value in initialization.as_dict().items()
            }
        )

    def forward(self) -> dict[str, Tensor]:
        return {name: value.clamp_min(0.0) for name, value in self.values.items()}

    def as_dict(self) -> dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self().items()}


def supervised_contrastive_loss(
    embeddings: Tensor, labels: Tensor, temperature: float = 0.07
) -> Tensor:
    """Outer-log supervised contrastive loss over same-emotion positives."""
    normalized = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
    similarities = normalized @ normalized.T / temperature
    batch_size = embeddings.shape[0]
    self_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    nonself = ~self_mask
    logits = similarities.masked_fill(self_mask, -torch.inf)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive = labels.reshape(-1, 1).eq(labels.reshape(1, -1)) & nonself
    positive_count = positive.sum(dim=1)
    eligible = positive_count > 0
    if not eligible.any():
        return embeddings.sum() * 0.0
    positive_log_probabilities = log_probabilities.masked_fill(~positive, 0.0)
    per_anchor = -positive_log_probabilities.sum(dim=1) / positive_count.clamp_min(1)
    return per_anchor[eligible].mean()


def cross_subject_prototype_loss(
    embeddings: Tensor,
    labels: Tensor,
    subject_ids: Tensor,
    num_classes: int,
    inter_margin: float = 0.20,
) -> Tensor:
    """Align minibatch subject-emotion prototypes with global class prototypes."""
    normalized = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
    class_prototypes = []
    intra_terms = []
    for class_index in range(num_classes):
        class_mask = labels == class_index
        if int(class_mask.sum()) < 2:
            continue
        class_prototype = F.normalize(
            normalized[class_mask].mean(dim=0), p=2, dim=0, eps=1e-12
        )
        class_prototypes.append(class_prototype)
        for subject in torch.unique(subject_ids[class_mask]):
            group_mask = class_mask & subject_ids.eq(subject)
            subject_prototype = F.normalize(
                normalized[group_mask].mean(dim=0), p=2, dim=0, eps=1e-12
            )
            intra_terms.append(1.0 - torch.dot(subject_prototype, class_prototype))

    zero = embeddings.sum() * 0.0
    loss_intra = torch.stack(intra_terms).mean() if intra_terms else zero
    if len(class_prototypes) < 2:
        return loss_intra
    prototypes = torch.stack(class_prototypes)
    similarities = prototypes @ prototypes.T
    off_diagonal = ~torch.eye(
        len(class_prototypes), device=embeddings.device, dtype=torch.bool
    )
    loss_inter = F.relu(similarities[off_diagonal] - inter_margin).mean()
    return loss_intra + 0.5 * loss_inter


def compute_losses(
    outputs: dict[str, Tensor],
    labels: Tensor,
    subject_ids: Tensor,
    num_classes: int,
    weights: LearnableLossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute Eq.40; six auxiliary coefficients receive autograd gradients."""
    probabilities = F.softmax(outputs["classical_logits"].detach(), dim=-1)
    residual_target = (
        F.one_hot(labels, num_classes).to(probabilities.dtype) - probabilities
    )
    terms = {
        "emotion": F.cross_entropy(outputs["emotion_logits"], labels),
        "classical": F.cross_entropy(outputs["classical_logits"], labels),
        "frontend": F.cross_entropy(outputs["frontend_logits"], labels),
        "subject": F.cross_entropy(outputs["subject_logits"], subject_ids),
        "supcon": supervised_contrastive_loss(
            outputs["embedding"], labels, temperature=0.07
        ),
        "prototype": cross_subject_prototype_loss(
            outputs["embedding"], labels, subject_ids, num_classes
        ),
        "quantum": F.mse_loss(outputs["quantum_auxiliary_logits"], residual_target),
    }
    coefficients = weights()
    total = terms["emotion"] + sum(
        coefficients[name] * terms[name] for name in coefficients
    )
    return total, terms
