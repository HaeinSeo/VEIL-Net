from __future__ import annotations

import math

import torch
from torch import nn


class SurfacePatchRefiner(nn.Module):
    """Generate local 2D surface patches around learned coarse points."""

    def __init__(self, hidden_dim: int, output_points: int, patches: int = 256) -> None:
        super().__init__()
        self.output_points = output_points
        self.patches = min(patches, output_points)
        self.per_patch = math.ceil(output_points / self.patches)
        side = max(2, math.ceil(math.sqrt(self.per_patch)))
        axis = torch.linspace(-1, 1, side)
        u, v = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid", torch.stack([u.flatten(), v.flatten()], dim=-1)[:self.per_patch])
        self.patch_code = nn.Embedding(self.patches, 16)
        self.fold = nn.Sequential(
            nn.Linear(hidden_dim + 3 + 2 + 16, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3),
        )
        nn.init.normal_(self.fold[-1].weight, std=0.001)
        nn.init.zeros_(self.fold[-1].bias)

    def forward(self, generated: torch.Tensor, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = generated.shape[0]
        indices = torch.linspace(0, generated.shape[1] - 1, self.patches, device=generated.device).long()
        coarse = generated[:, indices]
        centers = coarse[:, :, None, :].expand(-1, -1, self.per_patch, -1)
        grid = self.grid.to(dtype=generated.dtype)[None, None].expand(batch, self.patches, -1, -1)
        codes = self.patch_code.weight[None, :, None].expand(batch, -1, self.per_patch, -1)
        context = feature[:, None, None].expand(-1, self.patches, self.per_patch, -1)
        offsets = self.fold(torch.cat([context, centers, grid, codes], dim=-1))
        points = (centers + offsets).reshape(batch, -1, 3)[:, :self.output_points]
        return points, coarse
