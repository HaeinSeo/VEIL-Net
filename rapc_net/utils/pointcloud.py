from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def validate_points(points: np.ndarray, name: str = "points") -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def normalize_unit_sphere(points: np.ndarray) -> tuple[np.ndarray, dict[str, list[float] | float]]:
    arr = validate_points(points)
    center = arr.mean(axis=0)
    shifted = arr - center
    scale = float(np.linalg.norm(shifted, axis=1).max())
    if scale <= 0:
        scale = 1.0
    return shifted / scale, {"center": center.tolist(), "scale": scale}


def sample_points(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    arr = validate_points(points)
    if len(arr) == 0:
        raise ValueError("cannot sample an empty point cloud")
    rng = np.random.default_rng(seed)
    replace = len(arr) < count
    idx = rng.choice(len(arr), size=count, replace=replace)
    return arr[idx].astype(np.float32)


def depth_mask_to_points(depth: np.ndarray, mask: np.ndarray, intrinsics: dict[str, float], depth_scale: float) -> np.ndarray:
    depth = np.asarray(depth)
    mask = np.asarray(mask).astype(bool)
    if depth.shape != mask.shape:
        raise ValueError(f"depth and mask shapes differ: {depth.shape} vs {mask.shape}")
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if len(xs) == 0:
        raise ValueError("mask selects no valid depth pixels")
    z = depth[ys, xs].astype(np.float32) / float(depth_scale)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def transform_points(points: np.ndarray, matrix_4x4: np.ndarray) -> np.ndarray:
    arr = validate_points(points)
    mat = np.asarray(matrix_4x4, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"matrix must have shape [4, 4], got {mat.shape}")
    homog = np.concatenate([arr, np.ones((len(arr), 1), dtype=np.float32)], axis=1)
    return (homog @ mat.T)[:, :3].astype(np.float32)


def random_rotation_z(points: np.ndarray, seed: int) -> np.ndarray:
    arr = validate_points(points)
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(-math.pi, math.pi))
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return arr @ rot.T
