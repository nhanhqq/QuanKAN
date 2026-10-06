"""KAN-linear layers used by the classical and quantum readouts."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise
        step = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1)
        knots = knots * step + grid_range[0]
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = torch.rand(
                self.grid_size + 1,
                self.in_features,
                self.out_features,
                device=self.grid.device,
                dtype=self.grid.dtype,
            )
            noise = (noise - 0.5) * self.scale_noise / self.grid_size
            interior_grid = self.grid.T[self.spline_order : -self.spline_order]
            coefficients = self._curve_to_coefficients(interior_grid, noise)
            self.spline_weight.copy_(self.scale_spline * coefficients)
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x: Tensor) -> Tensor:
        knots = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= knots[:, :-1]) & (x < knots[:, 1:])).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            left = (x - knots[:, : -(order + 1)]) / (
                knots[:, order:-1] - knots[:, : -(order + 1)] + 1e-8
            )
            right = (knots[:, order + 1 :] - x) / (
                knots[:, order + 1 :] - knots[:, 1:-order] + 1e-8
            )
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def _curve_to_coefficients(self, x: Tensor, y: Tensor) -> Tensor:
        basis = self.b_splines(x).transpose(0, 1)
        targets = y.transpose(0, 1)
        solution = torch.linalg.lstsq(basis, targets).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flattened = x.reshape(-1, self.in_features)
        base = F.linear(F.silu(flattened), self.scale_base * self.base_weight)
        spline_weights = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = F.linear(
            self.b_splines(flattened).reshape(flattened.size(0), -1),
            spline_weights.reshape(self.out_features, -1),
        )
        return (base + spline).reshape(*original_shape[:-1], self.out_features)
