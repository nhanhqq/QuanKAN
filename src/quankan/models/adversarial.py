"""Gradient reversal used for source-subject adversarial regularization."""

from __future__ import annotations

import torch
from torch import Tensor


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, strength: float) -> Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return -ctx.strength * gradient, None


def grad_reverse(x: Tensor, strength: float) -> Tensor:
    return GradientReversal.apply(x, strength)
