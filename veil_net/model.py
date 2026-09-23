"""Geometry-conditioned model defaults and portable, strict weight loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from rapc_net.models import RAPCNet, RAPCNetConfig


@dataclass(frozen=True)
class VEILNetConfig(RAPCNetConfig):
    """Architecture used by the geometry-query and coverage-balance runs."""

    bbox_margin: float = 0.0
    uncertainty_moves_points: bool = False
    use_spatial_transformer: bool = True
    input_frame_normalization: bool = True
    two_stage_folding: bool = True
    robust_input_geometry: bool = True
    geometry_queries: bool = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VEILNet(RAPCNet):
    """VEIL-Net, retaining the original state-dict keys for exact compatibility.

    ``forward`` accepts floating-point XYZ tensors of shape [B, N, 3] and
    returns ``completed_points`` in the input coordinate frame. Use ``eval``
    and ``torch.inference_mode`` for inference; no target surface is required.
    """

    def __init__(self, config: RAPCNetConfig | None = None) -> None:
        super().__init__(config if config is not None else VEILNetConfig())

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "cpu") -> VEILNet:
        """Load an original training checkpoint using its recorded configuration."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not {"model", "model_config"} <= state.keys():
            raise ValueError("Checkpoint must contain model and model_config mappings")
        # Older checkpoints must keep RAPC defaults for fields absent at training time.
        model = cls(RAPCNetConfig(**state["model_config"]))
        model.load_state_dict(state["model"], strict=True)
        model.artifact_metadata = {"checkpoint_sha256": sha256_file(path)}
        return model.to(device).eval()

    def save_pretrained(self, directory: str | Path) -> Path:
        """Write config.json and model.safetensors into a new or empty directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError(f"Export directory must be empty: {directory}")
        config = {
            "format_version": 1,
            "architecture": "VEILNet",
            "model_config": asdict(self.config),
        }
        tensors = {
            key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()
        }
        save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
        (directory / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        return directory

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        device: str = "cpu",
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> VEILNet:
        """Load a local export or a Hugging Face model ID; never execute remote code."""
        directory = Path(source)
        if directory.is_dir():
            config_path, weights_path = directory / "config.json", directory / "model.safetensors"
        else:
            if isinstance(source, Path) or directory.is_absolute() or str(source).startswith("."):
                raise FileNotFoundError(f"Model directory not found: {source}")
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError('Hub loading requires: pip install "veil-net[hub]"') from exc
            options = {
                "repo_id": str(source),
                "revision": revision,
                "local_files_only": local_files_only,
            }
            config_path = Path(hf_hub_download(filename="config.json", **options))
            weights_path = Path(hf_hub_download(filename="model.safetensors", **options))
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if data.get("format_version") != 1 or data.get("architecture") != "VEILNet":
            raise ValueError("Unsupported VEIL-Net export format or architecture")
        model = cls(RAPCNetConfig(**data["model_config"]))
        model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        model.artifact_metadata = {
            "weights_sha256": sha256_file(weights_path),
            "config_sha256": sha256_file(config_path),
            "requested_revision": revision,
        }
        return model.to(device).eval()
