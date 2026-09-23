from __future__ import annotations

import numpy as np

from rapc_net.utils.pointcloud import sample_points, validate_points


DEGRADATIONS = ["clean", "dropout", "occlusion", "sparsity", "axial_noise", "boundary_corruption", "quantization", "mixed"]
SEVERITIES = ["mild", "medium", "severe"]


def degrade(points: np.ndarray, kind: str, severity: str, seed: int) -> tuple[np.ndarray, dict[str, float | str | int]]:
    pts = validate_points(points)
    rng = np.random.default_rng(seed)
    level = {"clean": 0.0, "mild": 0.15, "medium": 0.35, "severe": 0.6}.get(severity)
    if level is None:
        raise ValueError(f"unknown severity: {severity}")
    if kind == "clean":
        return pts.copy(), {"kind": kind, "severity": "clean", "seed": seed}
    out = pts.copy()
    if kind in {"dropout", "mixed"}:
        keep = rng.random(len(out)) > level
        out = out[keep] if keep.any() else out[:1]
    if kind in {"occlusion", "mixed"}:
        axis = int(rng.integers(0, 3))
        cutoff = float(np.quantile(out[:, axis], level))
        keep = out[:, axis] >= cutoff
        out = out[keep] if keep.any() else out[:1]
    if kind in {"sparsity", "mixed"}:
        target = max(16, int(len(out) * (1.0 - level)))
        out = sample_points(out, target, seed)
    if kind in {"axial_noise", "mixed"}:
        out[:, 2] += rng.normal(0.0, 0.005 * (1 + 4 * level), size=len(out)).astype(np.float32)
    if kind in {"boundary_corruption", "mixed"}:
        center = out.mean(axis=0)
        radius = np.linalg.norm(out - center, axis=1)
        boundary = radius > np.quantile(radius, 1.0 - min(level, 0.5))
        out[boundary] += rng.normal(0.0, 0.01 * (1 + level), size=(boundary.sum(), 3)).astype(np.float32)
    if kind in {"quantization", "mixed"}:
        q = 0.002 * (1 + 8 * level)
        out = np.round(out / q) * q
    if kind not in DEGRADATIONS:
        raise ValueError(f"unknown degradation: {kind}")
    return out.astype(np.float32), {"kind": kind, "severity": severity, "seed": seed, "level": level}
