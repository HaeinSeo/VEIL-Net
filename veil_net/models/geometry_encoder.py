from __future__ import annotations

import torch
from torch import nn


class GeometryEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.point_mlp(points)
        global_feature = self.global_mlp(features.max(dim=1).values)
        return features, global_feature
