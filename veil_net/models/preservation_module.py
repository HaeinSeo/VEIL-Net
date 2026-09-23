from __future__ import annotations

import torch
from torch import nn


class ObservedSurfacePreservation(nn.Module):
    def __init__(self, output_points: int = 16384) -> None:
        super().__init__()
        self.output_points = output_points

    def forward(
        self, partial: torch.Tensor, generated: torch.Tensor, risk: torch.Tensor
    ) -> torch.Tensor:
        keep_count = min(partial.shape[1], self.output_points // 4)
        reliable = torch.topk(1.0 - risk, k=keep_count, dim=1).indices
        kept = torch.gather(partial, 1, reliable.unsqueeze(-1).expand(-1, -1, 3))
        remaining = self.output_points - keep_count
        return torch.cat([kept, generated[:, :remaining, :]], dim=1)
