from __future__ import annotations

from pathlib import Path

from adapters.common import ExternalBaselineAdapter


class PoinTrAdapter(ExternalBaselineAdapter):
    candidate_modules = ("models.PoinTr", "models.pointr", "pointr")

    def __init__(self, repo_path: Path, output_points: int = 16384) -> None:
        super().__init__("PoinTr", repo_path, output_points)
