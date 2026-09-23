from __future__ import annotations

import torch
from torch import nn


class RiskGuidedDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, output_points: int = 16384) -> None:
        super().__init__()
        self.output_points = output_points
        self.seed = nn.Parameter(torch.randn(output_points, 3) * 0.02)
        self.risk_embed = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(
        self, routed_feature: torch.Tensor, risk: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch = routed_feature.shape[0]
        if risk is not None:
            risk_stats = torch.stack(
                [
                    risk.mean(dim=1),
                    risk.std(dim=1),
                    risk.max(dim=1).values,
                    (risk > 0.5).float().mean(dim=1),
                ],
                dim=1,
            )
            routed_feature = routed_feature + self.risk_embed(risk_stats)
        seed = self.seed.unsqueeze(0).expand(batch, -1, -1)
        feature = routed_feature.unsqueeze(1).expand(-1, self.output_points, -1)
        return seed + self.net(torch.cat([feature, seed], dim=-1))
