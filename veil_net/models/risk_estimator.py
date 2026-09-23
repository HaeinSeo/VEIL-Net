from __future__ import annotations

import torch
from torch import nn


class RiskEstimator(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        return self.logits(point_features).sigmoid()

    def logits(self, point_features: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[1](self.net[0](point_features))).squeeze(-1)
