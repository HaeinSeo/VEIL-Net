from __future__ import annotations

import torch
from torch import nn


class QualityRouter(nn.Module):
    def __init__(self, hidden_dim: int = 256, adapters: int = 4) -> None:
        super().__init__()
        self.adapters = adapters
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, adapters),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(adapters)
            ]
        )

    def forward(
        self, global_feature: torch.Tensor, risk: torch.Tensor, uniform: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quality = torch.stack(
            [
                risk.mean(dim=1),
                risk.std(dim=1),
                risk.max(dim=1).values,
                (risk > 0.5).float().mean(dim=1),
            ],
            dim=1,
        )
        if uniform:
            weights = global_feature.new_full(
                (global_feature.shape[0], self.adapters), 1.0 / self.adapters
            )
        else:
            weights = torch.softmax(self.router(torch.cat([global_feature, quality], dim=1)), dim=1)
        expert_outputs = torch.stack([expert(global_feature) for expert in self.experts], dim=1)
        routed = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return global_feature + routed, weights
