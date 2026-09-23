from __future__ import annotations

from pathlib import Path

from adapters.common import ExternalBaselineAdapter


class SnowflakeNetAdapter(ExternalBaselineAdapter):
    candidate_modules = ("models.snowflakenet", "models.SnowflakeNet", "snowflakenet")

    def __init__(self, repo_path: Path, output_points: int = 16384) -> None:
        super().__init__("SnowflakeNet", repo_path, output_points)
