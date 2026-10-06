"""Torch datasets for EEG feature and subject-label batches."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class EEGSubjectDataset(EEGDataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray,
        transform=None,
    ) -> None:
        super().__init__(features, labels)
        self.subject_ids = torch.as_tensor(subject_ids, dtype=torch.long)
        self.transform = transform

    def __getitem__(self, index: int):
        features = self.features[index]
        if self.transform is not None:
            features = self.transform(features)
        return features, self.labels[index], self.subject_ids[index]
