from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rapc_net.models.geometry_encoder import GeometryEncoder
from rapc_net.models.preservation_module import ObservedSurfacePreservation
from rapc_net.models.quality_router import QualityRouter
from rapc_net.models.risk_decoder import RiskGuidedDecoder
from rapc_net.models.risk_estimator import RiskEstimator
from rapc_net.models.uncertainty_gate import UncertaintyGate
from rapc_net.models.surface_refiner import SurfacePatchRefiner
from rapc_net.models.spatial_completion import SpatialCompletion
from rapc_net.models.input_geometry import screen_and_resample


@dataclass(frozen=True)
class RAPCNetConfig:
    input_points: int = 2048
    output_points: int = 16384
    hidden_dim: int = 256
    adapters: int = 4
    use_risk: bool = True
    use_router: bool = True
    use_preservation: bool = True
    use_uncertainty: bool = True
    uniform_routing: bool = False
    oracle_risk: bool = False
    preserve_exact_observed: bool = True
    bbox_margin: float = 0.20
    uncertainty_moves_points: bool = True
    use_surface_refiner: bool = False
    use_spatial_transformer: bool = False
    input_frame_normalization: bool = False
    two_stage_folding: bool = False
    robust_input_geometry: bool = False
    patch_grid_residual: bool = False
    geometry_queries: bool = False
    query_position_conditioning: bool = True
    spatial_tokens: int = 128
    spatial_patches: int = 256
    sparse_input_guard: bool = False
    sparse_unique_tokens: bool = False
    stable_input_frame: bool = False


class RAPCNet(nn.Module):
    def __init__(self, config: RAPCNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or RAPCNetConfig()
        self.encoder = GeometryEncoder(self.config.hidden_dim)
        self.risk = RiskEstimator(self.config.hidden_dim)
        self.router = QualityRouter(self.config.hidden_dim, self.config.adapters)
        self.decoder = None if self.config.use_spatial_transformer else RiskGuidedDecoder(self.config.hidden_dim, self.config.output_points)
        self.preserve = ObservedSurfacePreservation(self.config.output_points)
        self.gate = UncertaintyGate(self.config.hidden_dim)
        self.surface_refiner = SurfacePatchRefiner(self.config.hidden_dim, self.config.output_points) if self.config.use_surface_refiner and not self.config.use_spatial_transformer else None
        self.spatial_completion = SpatialCompletion(self.config.hidden_dim, self.config.output_points,
            tokens=self.config.spatial_tokens, patches=self.config.spatial_patches,
            two_stage=self.config.two_stage_folding, grid_residual=self.config.patch_grid_residual,
            geometry_queries=self.config.geometry_queries,
            query_position_conditioning=self.config.query_position_conditioning,
            sparse_unique_tokens=self.config.sparse_unique_tokens) if self.config.use_spatial_transformer else None
        if self.config.two_stage_folding and not self.config.use_spatial_transformer:
            raise ValueError('two_stage_folding requires use_spatial_transformer')
        if self.config.robust_input_geometry and not self.config.input_frame_normalization:
            raise ValueError('robust_input_geometry requires input_frame_normalization')
        if self.config.patch_grid_residual and not self.config.use_spatial_transformer:
            raise ValueError('patch_grid_residual requires use_spatial_transformer')
        if self.config.geometry_queries and not self.config.use_spatial_transformer:
            raise ValueError('geometry_queries requires use_spatial_transformer')
        if self.config.spatial_tokens < 1 or self.config.spatial_patches < 1:
            raise ValueError('Spatial token/patch counts must be positive')
        if self.config.sparse_input_guard and not self.config.robust_input_geometry:
            raise ValueError('sparse_input_guard requires robust_input_geometry')
        if self.config.sparse_unique_tokens and not self.config.use_spatial_transformer:
            raise ValueError('sparse_unique_tokens requires use_spatial_transformer')

    def forward(self, partial: torch.Tensor, risk_target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if partial.ndim != 3 or partial.shape[-1] != 3:
            raise ValueError(f"partial must have shape [B, N, 3], got {partial.shape}")
        source_indices = inlier_mask = None
        if self.config.robust_input_geometry:
            partial, source_indices, inlier_mask = screen_and_resample(partial, self.config.sparse_input_guard)
            if risk_target is not None:
                risk_target = torch.gather(risk_target, 1, source_indices)
        original_partial = partial
        frame_center = frame_scale = None
        if self.config.input_frame_normalization:
            # Input-only similarity frame. GT and evaluation coordinates are unchanged.
            with torch.no_grad():
                frame_center = partial.float().median(1, keepdim=True).values
                radii = torch.linalg.vector_norm(partial.float() - frame_center, dim=-1)
                quantile = 0.5 if self.config.robust_input_geometry else 0.9
                frame_scale = torch.quantile(radii, quantile, dim=1).clamp_min(1e-6)[:, None, None]
                if self.config.stable_input_frame:
                    # Bound input amplification when duplicate points collapse the median radius.
                    # This uses input geometry only, never GT or prediction clipping.
                    floor = radii.amax(dim=1)[:, None, None] / 32.0
                    frame_scale = torch.maximum(frame_scale, floor)
            partial = (partial - frame_center) / frame_scale
        point_features, global_feature = self.encoder(partial)
        risk_logits = self.risk.logits(point_features) if self.config.use_risk else None
        predicted_risk = risk_logits.sigmoid() if risk_logits is not None else partial.new_zeros(partial.shape[:2])
        point_risk = risk_target if self.config.oracle_risk and risk_target is not None else predicted_risk
        routed, route_weights = self.router(global_feature, point_risk, uniform=self.config.uniform_routing) if self.config.use_router else (global_feature, partial.new_zeros((partial.shape[0], self.config.adapters)))
        coarse = None
        if self.spatial_completion is not None:
            generated, coarse = self.spatial_completion(partial, point_features, routed, point_risk)
        else:
            generated = self.decoder(routed, point_risk if self.config.use_risk else None)
            if self.surface_refiner is not None:
                generated, coarse = self.surface_refiner(generated, routed)
        preserve_count = min(partial.shape[1], self.config.output_points // 4) if self.config.use_preservation else 0
        preserved = self.preserve(partial, generated, point_risk) if self.config.use_preservation else generated
        if self.config.use_uncertainty:
            completed, uncertainty = self.gate(routed, preserved)
            if not self.config.uncertainty_moves_points:
                completed = preserved
        else:
            completed = preserved
            uncertainty = partial.new_zeros((partial.shape[0], completed.shape[1]))
        if self.config.preserve_exact_observed and preserve_count > 0:
            completed = torch.cat([preserved[:, :preserve_count, :], completed[:, preserve_count:, :]], dim=1)
            uncertainty = torch.cat([uncertainty[:, :preserve_count].new_zeros(uncertainty[:, :preserve_count].shape), uncertainty[:, preserve_count:]], dim=1)
        if self.config.bbox_margin > 0:
            lower = partial.min(dim=1, keepdim=True).values - self.config.bbox_margin
            upper = partial.max(dim=1, keepdim=True).values + self.config.bbox_margin
            completed = torch.minimum(torch.maximum(completed, lower), upper)
        outputs = {
            "completed_points": completed,
            "generated_points": generated,
            "point_risk": point_risk,
            "predicted_risk": predicted_risk,
            "uncertainty": uncertainty,
            "point_uncertainty": uncertainty,
            "gate_score": 1.0 - uncertainty,
            "global_confidence": 1.0 - uncertainty.mean(dim=1),
            "route_weights": route_weights,
        }
        if coarse is not None:
            outputs["coarse_points"] = coarse
        if risk_logits is not None:
            outputs['risk_logits'] = risk_logits
        if frame_center is not None:
            for key in ('completed_points', 'generated_points', 'coarse_points'):
                if key in outputs:
                    outputs[key] = outputs[key] * frame_scale + frame_center
            if self.config.preserve_exact_observed and preserve_count > 0:
                indices = torch.topk(1.0 - point_risk, k=preserve_count, dim=1).indices
                kept = torch.gather(original_partial, 1, indices[..., None].expand(-1, -1, 3))
                outputs['completed_points'] = torch.cat((kept, outputs['completed_points'][:, preserve_count:]), 1)
        if source_indices is not None:
            outputs['input_source_indices'] = source_indices
            outputs['input_inlier_mask'] = inlier_mask
            outputs['processed_partial'] = original_partial
        if self.config.patch_grid_residual:
            patch_count = self.spatial_completion.patches
            retained = (self.config.output_points-preserve_count) // patch_count * patch_count
            outputs['patch_points'] = outputs['generated_points'][:, :retained].reshape(
                partial.shape[0], -1, patch_count, 3).transpose(1, 2)
        return outputs
