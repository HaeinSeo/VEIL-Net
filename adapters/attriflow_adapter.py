from __future__ import annotations

from pathlib import Path

from adapters.common import ExternalBaselineAdapter


class AttriFlowPCAdapter(ExternalBaselineAdapter):
    candidate_modules = ("models.PC_3DAttriFlow", "PC_3DAttriFlow")

    def __init__(self, repo_path: Path, output_points: int = 16384) -> None:
        super().__init__("3DAttriFlow-PC", repo_path, output_points)
