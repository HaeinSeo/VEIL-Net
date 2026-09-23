from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class BaselineInfo:
    name: str
    path: Path
    available: bool
    reason: str


class BaselineUnavailable(RuntimeError):
    pass


class ExternalBaselineAdapter(nn.Module):
    candidate_modules: tuple[str, ...] = ()

    def __init__(self, name: str, repo_path: Path, output_points: int = 16384) -> None:
        super().__init__()
        self.name = name
        self.repo_path = repo_path
        self.output_points = output_points
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        if not self.repo_path.exists():
            raise BaselineUnavailable(f"{self.name} repository path does not exist: {self.repo_path}")
        sys.path.insert(0, str(self.repo_path))
        errors: list[str] = []
        for module_name in self.candidate_modules:
            try:
                module = importlib.import_module(module_name)
                for attr in ("Model", "Net", "GRNet", "SnowflakeNet", "PoinTr"):
                    if hasattr(module, attr):
                        model = getattr(module, attr)()
                        if isinstance(model, nn.Module):
                            return model
                errors.append(f"{module_name}: imported but no known nn.Module class")
            except Exception as exc:
                errors.append(f"{module_name}: {exc}")
        raise BaselineUnavailable(f"Could not construct {self.name} from {self.repo_path}. Tried: {'; '.join(errors)}")

    def forward(self, partial: torch.Tensor) -> torch.Tensor:
        result = self.model(partial)
        if isinstance(result, dict):
            for key in ("completed_points", "coarse", "fine", "pred"):
                if key in result:
                    return result[key]
        if isinstance(result, (tuple, list)):
            return result[-1]
        if torch.is_tensor(result):
            return result
        raise RuntimeError(f"{self.name} returned unsupported output type: {type(result)!r}")
