from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rapc_net.utils.pointcloud import validate_points


@dataclass(frozen=True)
class CompletionMetrics:
    cd_l1: float
    cd_l2: float
    f_score: float
    f_score_001: float
    f_score_003: float
    f_score_005: float
    f_score_010: float
    f_score_020: float
    f_score_050: float
    accuracy: float
    completeness: float
    observed_region_error: float
    missing_region_cd: float
    normal_consistency: float
    dimension_mae: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def nearest_distances(src: np.ndarray, dst: np.ndarray, chunk: int = 4096) -> np.ndarray:
    src = validate_points(src, "src")
    dst = validate_points(dst, "dst")
    out = np.empty((len(src),), dtype=np.float32)
    for start in range(0, len(src), chunk):
        block = src[start : start + chunk]
        diff = block[:, None, :] - dst[None, :, :]
        out[start : start + chunk] = np.sqrt(np.sum(diff * diff, axis=2).min(axis=1))
    return out


def f_score_from_distances(pred_to_gt: np.ndarray, gt_to_pred: np.ndarray, threshold: float) -> float:
    precision = float((pred_to_gt < threshold).mean())
    recall = float((gt_to_pred < threshold).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def dimension_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_span = validate_points(pred).max(axis=0) - validate_points(pred).min(axis=0)
    gt_span = validate_points(gt).max(axis=0) - validate_points(gt).min(axis=0)
    return float(np.abs(pred_span - gt_span).mean())


def evaluate_completion(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray | None = None,
    f_threshold: float = 0.01,
) -> CompletionMetrics:
    pred = validate_points(prediction, "prediction")
    gt = validate_points(target, "target")
    pred_to_gt = nearest_distances(pred, gt)
    gt_to_pred = nearest_distances(gt, pred)
    cd_l1 = float(pred_to_gt.mean() + gt_to_pred.mean())
    cd_l2 = float((pred_to_gt**2).mean() + (gt_to_pred**2).mean())
    observed_error = float("nan")
    missing_cd = float("nan")
    if observed is not None:
        obs = validate_points(observed, "observed")
        observed_error = float(nearest_distances(obs, pred).mean())
        gt_to_obs = nearest_distances(gt, obs)
        missing = gt[gt_to_obs > f_threshold]
        if len(missing) > 0:
            missing_cd = float(nearest_distances(missing, pred).mean())
    return CompletionMetrics(
        cd_l1=cd_l1,
        cd_l2=cd_l2,
        f_score=f_score_from_distances(pred_to_gt, gt_to_pred, f_threshold),
        f_score_001=f_score_from_distances(pred_to_gt, gt_to_pred, 0.01),
        f_score_003=f_score_from_distances(pred_to_gt, gt_to_pred, 0.03),
        f_score_005=f_score_from_distances(pred_to_gt, gt_to_pred, 0.05),
        f_score_010=f_score_from_distances(pred_to_gt, gt_to_pred, 0.10),
        f_score_020=f_score_from_distances(pred_to_gt, gt_to_pred, 0.20),
        f_score_050=f_score_from_distances(pred_to_gt, gt_to_pred, 0.50),
        accuracy=float(pred_to_gt.mean()),
        completeness=float(gt_to_pred.mean()),
        observed_region_error=observed_error,
        missing_region_cd=missing_cd,
        normal_consistency=float("nan"),
        dimension_mae=dimension_mae(pred, gt),
    )


def _to_points_tensor(points: np.ndarray, device: str) -> torch.Tensor:
    arr = validate_points(points)
    return torch.from_numpy(arr.astype(np.float32, copy=False)).to(device)


def _torch_nearest_distances(src: torch.Tensor, dst: torch.Tensor, chunk: int) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for start in range(0, src.shape[0], chunk):
        block = src[start : start + chunk]
        dist = torch.cdist(block.unsqueeze(0), dst.unsqueeze(0), p=2).squeeze(0)
        values.append(dist.min(dim=1).values)
    return torch.cat(values, dim=0)


def _sample_for_eval(points: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
    arr = validate_points(points)
    if max_points is None or len(arr) <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=max_points, replace=False)
    return arr[idx]


def evaluate_completion_torch(
    prediction: np.ndarray,
    target: np.ndarray,
    observed: np.ndarray | None = None,
    f_threshold: float = 0.01,
    device: str = "cuda",
    chunk: int = 2048,
    max_points: int | None = None,
    seed: int = 0,
) -> CompletionMetrics:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    pred_np = _sample_for_eval(prediction, max_points, seed)
    gt_np = _sample_for_eval(target, max_points, seed + 1)
    pred = _to_points_tensor(pred_np, device)
    gt = _to_points_tensor(gt_np, device)
    with torch.no_grad():
        pred_to_gt = _torch_nearest_distances(pred, gt, chunk)
        gt_to_pred = _torch_nearest_distances(gt, pred, chunk)
        cd_l1 = pred_to_gt.mean() + gt_to_pred.mean()
        cd_l2 = (pred_to_gt.square()).mean() + (gt_to_pred.square()).mean()

        def f_score_at(threshold: float) -> torch.Tensor:
            precision = (pred_to_gt < threshold).float().mean()
            recall = (gt_to_pred < threshold).float().mean()
            return torch.where(precision + recall > 0, 2.0 * precision * recall / (precision + recall), precision.new_tensor(0.0))

        f_score = f_score_at(f_threshold)
        f_score_001 = f_score_at(0.01)
        f_score_003 = f_score_at(0.03)
        f_score_005 = f_score_at(0.05)
        f_score_010 = f_score_at(0.10)
        f_score_020 = f_score_at(0.20)
        f_score_050 = f_score_at(0.50)
        observed_error = pred.new_tensor(float("nan"))
        missing_cd = pred.new_tensor(float("nan"))
        if observed is not None:
            obs_np = _sample_for_eval(observed, max_points, seed + 2)
            obs = _to_points_tensor(obs_np, device)
            observed_error = _torch_nearest_distances(obs, pred, chunk).mean()
            gt_to_obs = _torch_nearest_distances(gt, obs, chunk)
            missing = gt[gt_to_obs > f_threshold]
            if missing.numel() > 0:
                missing_cd = _torch_nearest_distances(missing, pred, chunk).mean()
    return CompletionMetrics(
        cd_l1=float(cd_l1.detach().cpu()),
        cd_l2=float(cd_l2.detach().cpu()),
        f_score=float(f_score.detach().cpu()),
        f_score_001=float(f_score_001.detach().cpu()),
        f_score_003=float(f_score_003.detach().cpu()),
        f_score_005=float(f_score_005.detach().cpu()),
        f_score_010=float(f_score_010.detach().cpu()),
        f_score_020=float(f_score_020.detach().cpu()),
        f_score_050=float(f_score_050.detach().cpu()),
        accuracy=float(pred_to_gt.mean().detach().cpu()),
        completeness=float(gt_to_pred.mean().detach().cpu()),
        observed_region_error=float(observed_error.detach().cpu()),
        missing_region_cd=float(missing_cd.detach().cpu()),
        normal_consistency=float("nan"),
        dimension_mae=dimension_mae(pred_np, gt_np),
    )
