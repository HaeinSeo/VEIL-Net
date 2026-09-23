from __future__ import annotations

import numpy as np

from rapc_net.utils.pointcloud import validate_points


def rigid_align_svd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = validate_points(source, "source")
    dst = validate_points(target, "target")
    if len(src) != len(dst):
        raise ValueError("source and target must contain paired points with equal length")
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    h = (src - src_c).T @ (dst - dst_c)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    t = dst_c - r @ src_c
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = r.astype(np.float32)
    mat[:3, 3] = t.astype(np.float32)
    return mat


def rotation_error_deg(pred: np.ndarray, gt: np.ndarray) -> float:
    r = np.asarray(pred)[:3, :3] @ np.asarray(gt)[:3, :3].T
    cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def translation_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(pred)[:3, 3] - np.asarray(gt)[:3, 3]))
