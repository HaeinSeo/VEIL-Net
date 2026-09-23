from __future__ import annotations

import torch
from torch import nn


class UncertaintyGate(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, global_feature: torch.Tensor, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = global_feature.unsqueeze(1).expand(-1, points.shape[1], -1)
        uncertainty = self.net(torch.cat([feature, points], dim=-1))
        gate = 1.0 - uncertainty
        safe_points = points.mean(dim=1, keepdim=True).detach().expand_as(points)
        gated_points = points * gate + safe_points * (1.0 - gate)
        return gated_points, uncertainty.squeeze(-1)
