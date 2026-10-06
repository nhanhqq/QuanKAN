"""Four-qubit data-reuploading residual branch and measurement readout."""

from __future__ import annotations

import math

import pennylane as qml
import torch
from torch import Tensor, nn

from .kan import KANLinear


class QuantumBranch(nn.Module):
    def __init__(
        self, embedding_dim: int = 192, device_name: str = "default.qubit"
    ) -> None:
        super().__init__()
        self.num_qubits = 4
        self.num_layers = 4
        self.compression = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.angle_map = KANLinear(32, 4 * 4 * 2)
        self.device = qml.device(device_name, wires=self.num_qubits)
        self.rotation_weights = nn.Parameter(0.05 * torch.randn(4, 4, 3))

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(inputs: Tensor, weights: Tensor):
            for layer in range(4):
                for qubit in range(4):
                    qml.RY(inputs[:, layer, qubit, 0], wires=qubit)
                    qml.RZ(inputs[:, layer, qubit, 1], wires=qubit)
                    qml.Rot(*weights[layer, qubit], wires=qubit)
                if layer % 2 == 0:
                    for qubit in range(4):
                        qml.CNOT(wires=[qubit, (qubit + 1) % 4])
                else:
                    for qubit in range(4):
                        qml.CNOT(wires=[(qubit + 1) % 4, qubit])
            measurements = [qml.expval(qml.PauliZ(q)) for q in range(4)]
            measurements.extend(qml.expval(qml.PauliX(q)) for q in range(4))
            measurements.extend(
                qml.expval(qml.PauliZ(q) @ qml.PauliZ((q + 1) % 4)) for q in range(4)
            )
            return measurements

        self.circuit = circuit
        self.measurement_map = KANLinear(12, 32)

    def forward(self, x: Tensor) -> Tensor:
        angles = math.pi * torch.tanh(self.angle_map(self.compression(x)))
        angles = angles.reshape(-1, 4, 4, 2)
        measurements = self.circuit(angles.cpu(), self.rotation_weights.cpu())
        measurements = torch.stack(measurements, dim=1).to(
            device=x.device, dtype=x.dtype
        )
        return self.measurement_map(measurements)
