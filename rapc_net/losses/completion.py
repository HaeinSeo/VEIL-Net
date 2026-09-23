from __future__ import annotations

import torch
from torch import nn


def pairwise_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 3 or b.ndim != 3 or a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError(f"point tensors must be [B, N, 3] and [B, M, 3], got {a.shape}, {b.shape}")
    return torch.cdist(a, b, p=2)


def chamfer_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nearest_point_distances(pred, target).mean() + nearest_point_distances(target, pred).mean()


def nearest_point_distances(pred: torch.Tensor, target: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    # Find correspondences without retaining large pairwise matrices for backward.
    # Gradients through the selected point distances match nearest-neighbor loss.
    with torch.no_grad():
        indices = torch.cat([
            torch.cdist(pred[:, start:start + chunk].float(), target.float()).argmin(dim=2)
            for start in range(0, pred.shape[1], chunk)
        ], dim=1)
    nearest = torch.gather(target, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
    return torch.linalg.vector_norm(pred.float() - nearest.float(), dim=-1)


def surface_l2(pred: torch.Tensor, target: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    return nearest_point_distances(pred, target, chunk).square().mean()


def soft_fscore_loss(pred: torch.Tensor, target: torch.Tensor, thresholds=(0.2, 0.5)) -> torch.Tensor:
    pred_dist = nearest_point_distances(pred, target)
    gt_dist = nearest_point_distances(target, pred)
    losses = []
    for threshold in thresholds:
        temperature = threshold * 0.2
        precision = torch.sigmoid((threshold - pred_dist) / temperature).mean(dim=1)
        recall = torch.sigmoid((threshold - gt_dist) / temperature).mean(dim=1)
        losses.append(1.0 - 2.0 * precision * recall / (precision + recall).clamp_min(1e-8))
    return torch.stack(losses).mean()


def sampled_eval_fscore_loss(
    pred: torch.Tensor, target: torch.Tensor, max_points: int = 4096,
) -> torch.Tensor:
    """Training surrogate at raw-coordinate evaluation thresholds, not a metric.

    Fresh independent thinning discourages reliance on a particular output index
    or on dense clouds that will be subsampled by the unchanged evaluator.
    """
    if isinstance(max_points, bool) or int(max_points) != max_points or max_points < 1:
        raise ValueError('F-score point count must be a positive integer')
    max_points = int(max_points)
    pred = pred[:, torch.randperm(pred.shape[1], device=pred.device)[:max_points]]
    target = target[:, torch.randperm(target.shape[1], device=target.device)[:max_points]]
    return soft_fscore_loss(pred, target, thresholds=(0.03, 0.05))


def missing_region_l1(pred: torch.Tensor, target: torch.Tensor, partial: torch.Tensor, threshold: float = 0.03, per_sample: bool = False) -> torch.Tensor:
    with torch.no_grad():
        target_to_partial = nearest_point_distances(target, partial)
    target_to_pred = nearest_point_distances(target, pred)
    missing = target_to_partial > threshold
    if per_sample:
        counts = missing.sum(1)
        selected = (target_to_pred * missing).sum(1) / counts.clamp_min(1)
        return torch.where(counts > 0, selected, target_to_pred.mean(1)).mean()
    if not missing.any():
        return target_to_pred.mean()
    denom = missing.float().sum().clamp_min(1.0)
    return (target_to_pred * missing.float()).sum() / denom


def sampled_region_coverage(
    pred: torch.Tensor, target: torch.Tensor, partial: torch.Tensor,
    observed: torch.Tensor, max_points: int = 4096, threshold: float = 0.01,
    relative: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-only coverage after uniform thinning, with an original-unit mask.

    Exact copied inputs make full-cloud preservation loss zero. Random thinning
    requires nearby generated geometry to support those observations as well.
    Region membership uses the raw training input; preservation uses its existing
    input-only screened observations, so rejected noise is not a new target.
    """
    if max_points < 1 or threshold <= 0:
        raise ValueError('Coverage point count and threshold must be positive')

    def sample(points):
        if points.shape[1] <= max_points:
            return points
        indices = torch.randperm(points.shape[1], device=points.device)[:max_points]
        return points[:, indices]

    sampled_pred, sampled_target = sample(pred), sample(target)
    with torch.no_grad():
        missing = nearest_point_distances(sampled_target, partial) > threshold
        scale = torch.linalg.vector_norm(target.float().amax(1)-target.float().amin(1), dim=-1).clamp_min(1e-6) if relative else pred.new_ones(pred.shape[0])
    distances = nearest_point_distances(sampled_target, sampled_pred) / scale[:, None]
    counts = missing.sum(1)
    missing_loss = ((distances * missing).sum(1) / counts.clamp_min(1)).mean()
    observed_loss = (nearest_point_distances(observed, sampled_pred) / scale[:, None]).mean()
    return missing_loss, observed_loss


def local_repulsion(points: torch.Tensor, radius: float = 0.03, max_points: int = 1024) -> torch.Tensor:
    """PU-Net-inspired spacing hinge, not the paper's Gaussian energy.

    Subsample to bound memory; only nearby points are pushed apart. This does
    not supply missing geometry and must accompany target reconstruction loss.
    """
    if radius <= 0 or max_points < 2:
        raise ValueError("radius must be positive and max_points must be >= 2")
    if points.shape[1] < 2:
        return points.sum() * 0.0
    indices = torch.linspace(0, points.shape[1] - 1, min(max_points, points.shape[1]), device=points.device).long()
    sampled = points[:, indices].float()
    with torch.no_grad():
        distances = torch.cdist(sampled, sampled)
        distances.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        neighbors = distances.topk(min(4, sampled.shape[1] - 1), largest=False).indices
    batch = torch.arange(sampled.shape[0], device=points.device)[:, None, None]
    nearby = sampled[batch, neighbors]
    spacing = torch.linalg.vector_norm(sampled.unsqueeze(2) - nearby, dim=-1)
    return (1.0 - spacing / radius).clamp_min(0).square().mean()


class CompletionLoss(nn.Module):
    def __init__(
        self,
        lambda_cd: float = 1.0,
        lambda_missing: float = 0.0,
        lambda_preserve: float = 0.1,
        lambda_risk: float = 0.05,
        lambda_uncertainty: float = 0.01,
        lambda_uniform: float = 0.001,
        lambda_balance: float = 0.01,
        missing_threshold: float = 0.03,
        lambda_surface: float = 0.0,
        lambda_coarse: float = 0.0,
        lambda_soft_fscore: float = 0.0,
        missing_on_completed: bool = False,
        lambda_repulsion: float = 0.0,
        relative_geometry_loss: bool = False,
        lambda_patch_repulsion: float = 0.0,
        lambda_extent: float = 0.0,
        lambda_coverage_missing: float = 0.0,
        lambda_coverage_preserve: float = 0.0,
        coverage_points: int = 4096,
        coverage_threshold: float = 0.01,
        lambda_eval_fscore: float = 0.0,
        eval_fscore_points: int = 4096,
    ) -> None:
        super().__init__()
        self.weights = {
            "cd": lambda_cd,
            "missing": lambda_missing,
            "preserve": lambda_preserve,
            "risk": lambda_risk,
            "uncertainty": lambda_uncertainty,
            "uniform": lambda_uniform,
            "balance": lambda_balance,
            "surface": lambda_surface,
            "coarse": lambda_coarse,
            "soft_fscore": lambda_soft_fscore,
            "repulsion": lambda_repulsion,
            "patch_repulsion": lambda_patch_repulsion,
            "extent": lambda_extent,
            "coverage_missing": lambda_coverage_missing,
            "coverage_preserve": lambda_coverage_preserve,
            "eval_fscore": lambda_eval_fscore,
        }
        if int(coverage_points) != coverage_points or coverage_points < 1 or not coverage_threshold > 0:
            raise ValueError('Invalid coverage resolution or threshold')
        self.coverage_points = int(coverage_points)
        self.coverage_threshold = float(coverage_threshold)
        self.missing_threshold = float(missing_threshold)
        self.missing_on_completed = bool(missing_on_completed)
        self.relative_geometry_loss = bool(relative_geometry_loss)
        if isinstance(eval_fscore_points, bool) or int(eval_fscore_points) != eval_fscore_points or eval_fscore_points < 1:
            raise ValueError('F-score point count must be a positive integer')
        self.eval_fscore_points = int(eval_fscore_points)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = dict(outputs)
        target = batch["complete"]
        partial = outputs.get('processed_partial', batch.get("partial"))
        parts: dict[str, torch.Tensor] = {}
        # Apply before relative normalization: 0.03/0.05 are original-unit tolerances.
        if self.weights['eval_fscore'] > 0:
            parts['eval_fscore'] = sampled_eval_fscore_loss(
                outputs['completed_points'], target, self.eval_fscore_points,
            ) * self.weights['eval_fscore']
        if self.weights['coverage_missing'] > 0 or self.weights['coverage_preserve'] > 0:
            if partial is None or 'partial' not in batch:
                raise ValueError('Sampled coverage requires training input points')
            missing, observed = sampled_region_coverage(outputs['completed_points'], target,
                batch['partial'], partial, self.coverage_points, self.coverage_threshold,
                relative=self.relative_geometry_loss)
            if self.weights['coverage_missing'] > 0:
                parts['coverage_missing'] = missing * self.weights['coverage_missing']
            if self.weights['coverage_preserve'] > 0:
                parts['coverage_preserve'] = observed * self.weights['coverage_preserve']
        if self.relative_geometry_loss:
            # Training-only unit-diagonal loss frame. Network outputs and evaluation
            # stay in original coordinates; GT is never used for inference alignment.
            with torch.no_grad():
                center = target.float().mean(1, keepdim=True)
                scale = torch.linalg.vector_norm(target.float().amax(1)-target.float().amin(1), dim=-1).clamp_min(1e-6)[:, None, None]
            target = (target-center)/scale
            if partial is not None:
                partial = (partial-center)/scale
            for key in ('completed_points', 'generated_points', 'coarse_points'):
                if key in outputs:
                    outputs[key] = (outputs[key]-center)/scale
            if 'patch_points' in outputs:
                outputs['patch_points'] = (outputs['patch_points']-center[:, None])/scale[:, None]
        pred = outputs["completed_points"]
        total = pred.new_tensor(0.0)
        if self.weights['extent'] > 0:
            pred_extent = pred.amax(dim=1) - pred.amin(dim=1)
            target_extent = target.amax(dim=1) - target.amin(dim=1)
            parts['extent'] = (pred_extent-target_extent).abs().mean() * self.weights['extent']
        if self.weights['patch_repulsion'] > 0:
            if 'patch_points' not in outputs:
                raise ValueError('lambda_patch_repulsion requires patch_points')
            patches = outputs['patch_points']
            if patches.shape[2] < 2:
                raise ValueError('Patch repulsion needs at least two retained samples per patch')
            flat = patches.reshape(-1, patches.shape[2], 3)
            parts['patch_repulsion'] = local_repulsion(flat, radius=0.003) * self.weights['patch_repulsion']
        if self.weights["repulsion"] > 0:
            parts["repulsion"] = local_repulsion(pred) * self.weights["repulsion"]
        if self.weights["cd"] > 0:
            parts["cd"] = chamfer_l1(pred, target) * self.weights["cd"]
        if self.weights["surface"] > 0:
            parts["surface"] = surface_l2(pred, target) * self.weights["surface"]
        if self.weights["coarse"] > 0:
            if "coarse_points" not in outputs:
                raise ValueError("lambda_coarse requires a surface-refiner model")
            parts["coarse"] = chamfer_l1(outputs["coarse_points"], target) * self.weights["coarse"]
        if self.weights["soft_fscore"] > 0:
            thresholds = (0.01, 0.02) if self.relative_geometry_loss else (0.2, 0.5)
            parts["soft_fscore"] = soft_fscore_loss(pred, target, thresholds=thresholds) * self.weights["soft_fscore"]
        if partial is not None and self.weights["missing"] > 0:
            generated = pred if self.missing_on_completed else outputs.get("generated_points", pred)
            parts["missing"] = missing_region_l1(generated, target, partial, threshold=self.missing_threshold, per_sample=self.relative_geometry_loss) * self.weights["missing"]
        if partial is not None and self.weights["preserve"] > 0:
            dist = nearest_point_distances(partial, pred).mean()
            parts["preserve"] = dist * self.weights["preserve"]
        if self.weights["risk"] > 0 and ("point_risk" in outputs or "predicted_risk" in outputs):
            risk = outputs["predicted_risk"] if "predicted_risk" in outputs else outputs["point_risk"]
            target_risk = batch.get("risk_target")
            if target_risk is not None and 'input_source_indices' in outputs:
                target_risk = torch.gather(target_risk, 1, outputs['input_source_indices'])
            valid = torch.isfinite(target_risk) if target_risk is not None else None
            if valid is not None and valid.any():
                targets = target_risk[valid]
                if ((targets < 0) | (targets > 1)).any():
                    raise ValueError('Risk targets must be in [0, 1]')
                if 'risk_logits' in outputs:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs['risk_logits'][valid], targets)
                else:
                    loss = torch.nn.functional.smooth_l1_loss(risk[valid], targets)
                parts['risk'] = loss * self.weights['risk']
            else:
                # Missing labels must not encourage saturated 0/1 predictions.
                parts['risk'] = risk.sum() * 0.0
        if self.weights["uncertainty"] > 0 and "uncertainty" in outputs:
            unc = outputs["uncertainty"]
            with torch.no_grad():
                pred_to_gt = nearest_point_distances(pred, target)
                unc_target = pred_to_gt / (pred_to_gt.mean(dim=1, keepdim=True) + 1e-6)
                unc_target = unc_target.clamp(0.0, 1.0)
            parts["uncertainty"] = torch.nn.functional.smooth_l1_loss(unc, unc_target) * self.weights["uncertainty"]
        if self.weights["balance"] > 0 and "route_weights" in outputs:
            route = outputs["route_weights"]
            if route.numel() > 0:
                target = route.new_full((route.shape[1],), 1.0 / route.shape[1])
                parts["balance"] = (route.mean(dim=0) - target).square().sum() * self.weights["balance"]
        if self.weights["uniform"] > 0:
            centroid = pred.mean(dim=1, keepdim=True)
            radius = torch.norm(pred - centroid, dim=-1).std(dim=1).mean()
            parts["uniform"] = radius * self.weights["uniform"]
        for value in parts.values():
            total = total + value
        metrics = {k: float(v.detach().cpu()) for k, v in parts.items()}
        metrics.update({f"raw_{k}": metrics[k] / self.weights[k] for k in parts})
        return total, metrics | {"total": float(total.detach().cpu())}
