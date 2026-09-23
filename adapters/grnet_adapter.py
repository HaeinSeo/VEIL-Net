from __future__ import annotations

from pathlib import Path

import sys

import torch
from torch import nn

from adapters.common import BaselineUnavailable, ExternalBaselineAdapter


class GRNetAdapter(ExternalBaselineAdapter):
    candidate_modules = ("models.grnet", "model.grnet", "grnet", "models.GRNet")

    def __init__(self, repo_path: Path, output_points: int = 16384) -> None:
        super().__init__("GRNet", repo_path, output_points)

    def _load_model(self) -> nn.Module:
        if not self.repo_path.exists():
            raise BaselineUnavailable(f"GRNet repository path does not exist: {self.repo_path}")
        sys.path.insert(0, str(self.repo_path))
        try:
            from models.grnet import GRNet

            return GRNet(cfg=None)
        except Exception as exc:
            raise BaselineUnavailable(
                "Could not import/construct GRNet. Build GRNet extensions first: "
                "extensions/chamfer_dist, extensions/cubic_feature_sampling, extensions/gridding, extensions/gridding_loss."
            ) from exc

    def forward(self, partial: torch.Tensor) -> torch.Tensor:
        result = self.model({"partial_cloud": partial})
        if isinstance(result, (tuple, list)):
            return result[-1]
        if torch.is_tensor(result):
            return result
        raise RuntimeError(f"GRNet returned unsupported output type: {type(result)!r}")
