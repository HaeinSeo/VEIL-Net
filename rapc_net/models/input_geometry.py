"""Input-only radial outlier screening with explicit source-index provenance."""
import torch


def unique_point_indices(points: torch.Tensor) -> torch.Tensor:
    """Return source indices for exact unique coordinates, in coordinate order."""
    with torch.no_grad():
        unique, inverse = torch.unique(points, dim=0, return_inverse=True)
        first = torch.full((len(unique),), len(points), device=points.device, dtype=torch.long)
        first.scatter_reduce_(0, inverse, torch.arange(len(points), device=points.device), reduce='amin')
    return first


def screen_and_resample(points: torch.Tensor, sparse_input_guard: bool = False):
    if points.ndim != 3 or points.shape[1] == 0 or not torch.isfinite(points).all():
        raise ValueError('Expected nonempty finite [B, N, 3] input')
    with torch.no_grad():
        xyz = points.float()
        center = xyz.median(1, keepdim=True).values
        radius = torch.linalg.vector_norm(xyz-center, dim=-1)
        median = radius.median(1, keepdim=True).values
        mad = (radius-median).abs().median(1, keepdim=True).values
        cutoff = torch.maximum(4*median, median+6*mad).clamp_min(1e-6)
        inlier = radius <= cutoff
        if sparse_input_guard:
            # Repetition is not independent evidence for outlier removal.
            for i in range(len(xyz)):
                if len(unique_point_indices(xyz[i])) <= 16:
                    inlier[i] = True
        selected = []
        for mask in inlier:
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            # Maintain the network's fixed input count without moving any point.
            selected.append(indices[torch.arange(points.shape[1], device=points.device) % len(indices)])
        source = torch.stack(selected)
    cleaned = torch.gather(points, 1, source[..., None].expand(-1, -1, 3))
    return cleaned, source, inlier
