"""Input-only observation diagnostics; no GT-conditioned filtering."""
from __future__ import annotations

import numpy as np


def observation_stats(partial: np.ndarray) -> dict:
    points = np.asarray(partial, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 partial points")
    finite = points[np.isfinite(points).all(axis=1)]
    unique = np.unique(finite, axis=0)
    count = len(unique)
    # Exact duplicates are introduced by the cached fixed-count resampling.
    band = "very_sparse" if count <= 16 else "sparse" if count <= 128 else "supported"
    return {
        "input_count": len(points), "finite_count": len(finite),
        "unique_count": count, "unique_fraction": count / max(len(points), 1),
        "observation_band": band,
        "observation_action": "request_another_view" if count <= 16 else "evaluate_completion",
    }
