"""Input validation and deterministic sampling shared by inference and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rapc_net.utils.pointcloud import sample_points, validate_points
from veil_net.model import VEILNet


def read_cloud(path: str | Path, key: str = "partial") -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise ValueError(f"{path.name} has no {key!r} array; available: {archive.files}")
            points = archive[key].copy()
    else:
        raise ValueError("Use a .npy XYZ array or .npz archive")
    points = validate_points(points, key)
    if len(points) == 0:
        raise ValueError("Point clouds must not be empty")
    return points


def prepare_input(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    points = validate_points(points)
    if count < 1 or len(points) == 0:
        raise ValueError("Input point count must be positive")
    # Preserve already prepared benchmark pairs byte-for-byte and in their original order.
    return points if len(points) == count else sample_points(points, count, seed)


def predict(model: VEILNet, points: np.ndarray, *, seed: int = 0) -> dict[str, np.ndarray]:
    partial = prepare_input(points, model.config.input_points, seed)
    parameter = next(model.parameters())
    batch = torch.from_numpy(np.ascontiguousarray(partial)).to(parameter.device, parameter.dtype)[
        None
    ]
    with torch.inference_mode():
        output = model(batch)
    result = {key: value[0].detach().cpu().numpy() for key, value in output.items()}
    if not np.isfinite(result["completed_points"]).all():
        raise FloatingPointError("Model produced nonfinite surface coordinates")
    result["input_partial"] = partial
    result["prediction_complete"] = result.pop("completed_points")
    return result


def load_model(
    source: str, device: str, *, revision: str | None = None, local_files_only: bool = False
) -> VEILNet:
    if Path(source).suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return VEILNet.from_checkpoint(source, device=device)
    return VEILNet.from_pretrained(
        source, device=device, revision=revision, local_files_only=local_files_only
    )
