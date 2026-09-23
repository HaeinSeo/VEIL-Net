"""Spatial-token completion experiment using native PyTorch attention."""

import math

import torch
from torch import nn

from veil_net.models.input_geometry import unique_point_indices


def farthest_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    with torch.no_grad():
        xyz = points.float()
        count = min(count, xyz.shape[1])
        indices = torch.empty(xyz.shape[0], count, dtype=torch.long, device=xyz.device)
        distance = torch.full(xyz.shape[:2], float("inf"), device=xyz.device)
        current = (xyz - xyz.mean(1, keepdim=True)).square().sum(-1).argmax(1)
        batch = torch.arange(xyz.shape[0], device=xyz.device)
        for i in range(count):
            indices[:, i] = current
            distance = torch.minimum(distance, (xyz - xyz[batch, current, None]).square().sum(-1))
            current = distance.argmax(1)
        return indices


class SpatialCompletion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_points: int,
        tokens: int = 128,
        patches: int = 256,
        two_stage: bool = False,
        grid_residual: bool = False,
        geometry_queries: bool = False,
        sparse_unique_tokens: bool = False,
        query_position_conditioning: bool = True,
    ):
        super().__init__()
        self.tokens = tokens
        self.patches = min(patches, output_points)
        self.output_points = output_points
        self.per_patch = math.ceil(output_points / self.patches)
        self.geometry_queries = geometry_queries
        self.query_position_conditioning = query_position_conditioning
        self.sparse_unique_tokens = sparse_unique_tokens
        self.local = nn.Sequential(
            nn.Linear(hidden_dim + 4 + (7 if geometry_queries else 0), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, 4, hidden_dim * 2, dropout=0.0, batch_first=True, norm_first=True
            ),
            2,
            enable_nested_tensor=False,
        )
        self.queries = None if geometry_queries else nn.Embedding(self.patches, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden_dim, 4, hidden_dim * 2, dropout=0.0, batch_first=True, norm_first=True
            ),
            2,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.centers = None if geometry_queries else nn.Linear(hidden_dim, 3)
        if geometry_queries:
            self.geometry_global = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.center_prediction = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, self.patches * 3),
            )
            self.query_builder = nn.Sequential(
                nn.Linear(hidden_dim + 3, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        side = max(2, math.ceil(math.sqrt(self.per_patch)))
        u, v = torch.meshgrid(
            torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij"
        )
        self.register_buffer("grid", torch.stack((u.flatten(), v.flatten()), -1)[: self.per_patch])
        self.fold = nn.Sequential(
            nn.Linear(hidden_dim + 5, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.rebuild_feature = nn.Linear(hidden_dim * 2 + 3, hidden_dim) if two_stage else None
        self.fold_second = (
            nn.Sequential(
                nn.Linear(hidden_dim + 3, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 3),
            )
            if two_stage
            else None
        )
        self.patch_basis = nn.Linear(hidden_dim, 6) if grid_residual else None
        if self.patch_basis is not None:
            nn.init.normal_(self.patch_basis.weight, std=0.001)
            with torch.no_grad():
                self.patch_basis.bias.copy_(torch.tensor([0.02, 0.0, 0.0, 0.0, 0.02, 0.0]))

    def forward(self, partial, features, global_feature, risk):
        if partial.shape[1] == 0:
            raise ValueError("Spatial completion requires at least one observed point")
        if self.sparse_unique_tokens:
            unique_indices = [unique_point_indices(points) for points in partial]
            sparse = [len(idx) <= 16 and len(idx) < partial.shape[1] for idx in unique_indices]
            if any(sparse):
                # Keep the normal batched path unchanged when no sparse input exists.
                results = []
                for i, idx in enumerate(unique_indices):
                    if not sparse[i]:
                        idx = torch.arange(partial.shape[1], device=partial.device)
                    results.append(
                        self._forward_geometry(
                            partial[i : i + 1, idx],
                            features[i : i + 1, idx],
                            global_feature[i : i + 1],
                            risk[i : i + 1, idx],
                        )
                    )
                return tuple(torch.cat([result[j] for result in results], dim=0) for j in (0, 1))
        return self._forward_geometry(partial, features, global_feature, risk)

    def _forward_geometry(self, partial, features, global_feature, risk):
        indices = farthest_indices(partial, self.tokens)
        batch = torch.arange(partial.shape[0], device=partial.device)[:, None]
        anchors = partial[batch, indices]
        with torch.no_grad():
            neighbors = (
                torch.cdist(anchors.float(), partial.float())
                .topk(min(16, partial.shape[1]), largest=False)
                .indices
            )
        b = batch[:, :, None]
        relative = partial[b, neighbors] - anchors[:, :, None]
        local = torch.cat((features[b, neighbors], relative, risk[b, neighbors, None]), -1)
        if self.geometry_queries:
            # Unoriented normal, normalized covariance spectrum and local density.
            # Descriptors use observed neighborhoods only, never target geometry.
            with torch.no_grad():
                centered = relative.float() - relative.float().mean(2, keepdim=True)
                covariance = centered.transpose(-1, -2) @ centered / centered.shape[2]
                values, vectors = torch.linalg.eigh(covariance)
                normal = vectors[..., 0]
                axis = normal.abs().argmax(-1, keepdim=True)
                sign = normal.gather(-1, axis).sign()
                normal = normal * torch.where(sign == 0, torch.ones_like(sign), sign)
                spectrum = values.clamp_min(0) / values.clamp_min(0).sum(
                    -1, keepdim=True
                ).clamp_min(1e-8)
                radius = torch.linalg.vector_norm(relative.float(), dim=-1).amax(-1, keepdim=True)
                density = torch.log1p(1 / radius.clamp_min(1e-4))
                descriptor = torch.cat((normal, spectrum, density), -1)
            local = torch.cat(
                (local, descriptor[:, :, None].expand(-1, -1, local.shape[2], -1)), -1
            )
        memory = self.encoder(self.local(local).max(2).values + self.position(anchors))
        if self.geometry_queries:
            context = self.geometry_global(
                torch.cat((memory.max(1).values, memory.mean(1), global_feature), -1)
            )
            coarse = self.center_prediction(context).reshape(
                partial.shape[0], self.patches, 3
            ) + partial.mean(1, keepdim=True)
            # Isolate center-coordinate conditioning; preserve descriptors, centers and folding.
            query_centers = coarse if self.query_position_conditioning else torch.zeros_like(coarse)
            queries = self.query_builder(
                torch.cat((context[:, None].expand(-1, self.patches, -1), query_centers), -1)
            ) + self.position(query_centers)
            decoded = self.norm(self.decoder(queries, memory))
        else:
            queries = self.queries.weight[None] + global_feature[:, None]
            decoded = self.norm(self.decoder(queries, memory))
            coarse = self.centers(decoded) + partial.mean(1, keepdim=True)
        if self.rebuild_feature is not None:
            global_context = decoded.max(1, keepdim=True).values.expand_as(decoded)
            decoded = self.rebuild_feature(torch.cat((decoded, global_context, coarse), -1))
        context = decoded[:, :, None].expand(-1, -1, self.per_patch, -1)
        centers = coarse[:, :, None].expand(-1, -1, self.per_patch, -1)
        grid = self.grid.to(decoded.dtype)[None, None].expand(
            partial.shape[0], self.patches, -1, -1
        )
        offsets = self.fold(torch.cat((context, centers, grid), -1))
        if self.fold_second is not None:
            offsets = self.fold_second(torch.cat((context, offsets), -1))
        if self.patch_basis is not None:
            basis = self.patch_basis(decoded).reshape(partial.shape[0], self.patches, 2, 3)
            offsets = offsets + torch.einsum("bpsi,bpij->bpsj", grid, basis)
        points = centers + offsets
        # Interleave patches so preservation truncation does not drop whole patches.
        points = points.transpose(1, 2).reshape(partial.shape[0], -1, 3)[:, : self.output_points]
        return points, coarse
