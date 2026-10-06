import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LearnableLossWeights(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.values = nn.ParameterDict({
            "classical": nn.Parameter(torch.tensor(0.20)),
            "frontend": nn.Parameter(torch.tensor(0.25)),
            "subject": nn.Parameter(torch.tensor(0.20)),
            "supcon": nn.Parameter(torch.tensor(0.08)),
            "prototype": nn.Parameter(torch.tensor(0.05)),
            "quantum": nn.Parameter(torch.tensor(0.15)),
        })

    def forward(self) -> dict[str, Tensor]:
        return {name: parameter.clamp_min(0.0) for name, parameter in self.values.items()}


def supervised_contrastive_loss(
    embeddings: Tensor, labels: Tensor, temperature: float = 0.07
) -> Tensor:
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

    if intra_terms:
        loss_intra = torch.stack(intra_terms).mean()
    else:
        loss_intra = embeddings.sum() * 0.0

    if len(class_prototypes) >= 2:
        prototypes = torch.stack(class_prototypes)
        similarities = prototypes @ prototypes.T
        off_diagonal = ~torch.eye(
            len(class_prototypes), device=embeddings.device, dtype=torch.bool
        )
        loss_inter = F.relu(similarities[off_diagonal] - inter_margin).mean()
    else:
        loss_inter = embeddings.sum() * 0.0
    return loss_intra + 0.5 * loss_inter


def compute_losses(
    outputs: dict[str, Tensor],
    labels: Tensor,
    subject_ids: Tensor,
    weights: LearnableLossWeights,
    num_classes: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    criterion = nn.CrossEntropyLoss()
    probabilities = F.softmax(outputs["classical_logits"].detach(), dim=-1)
    residual_target = F.one_hot(labels, num_classes).to(probabilities.dtype) - probabilities
    terms = {
        "emotion": criterion(outputs["emotion_logits"], labels),
        "classical": criterion(outputs["classical_logits"], labels),
        "frontend": criterion(outputs["frontend_logits"], labels),
        "subject": criterion(outputs["subject_logits"], subject_ids),
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
