import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from veil_net import VEILNet, VEILNetConfig
from veil_net.inference import predict, prepare_input, read_cloud
from veil_net.models import CompletionConfig, CompletionNetwork


def tiny_model():
    return VEILNet(
        VEILNetConfig(
            input_points=16, output_points=32, hidden_dim=16, spatial_tokens=4, spatial_patches=4
        )
    ).eval()


def test_public_defaults_match_geometry_query_architecture():
    config = VEILNetConfig()
    assert config.geometry_queries and config.query_position_conditioning
    assert config.robust_input_geometry and config.two_stage_folding
    assert config.bbox_margin == 0 and not config.uncertainty_moves_points
    assert not CompletionConfig().geometry_queries


def test_checkpoint_and_portable_export_preserve_predictions(tmp_path):
    torch.set_num_threads(2)
    original = tiny_model()
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {"model_config": asdict(original.config), "model": original.state_dict()}, checkpoint
    )
    model = VEILNet.from_checkpoint(checkpoint)
    assert not model.training
    model.save_pretrained(tmp_path / "export")
    restored = VEILNet.from_pretrained(tmp_path / "export")
    points = np.random.default_rng(4).normal(size=(16, 3)).astype(np.float32)
    expected = predict(original, points)["prediction_complete"]
    np.testing.assert_array_equal(predict(model, points)["prediction_complete"], expected)
    np.testing.assert_array_equal(predict(restored, points)["prediction_complete"], expected)
    with pytest.raises(FileExistsError):
        model.save_pretrained(tmp_path / "export")


def test_legacy_defaults_not_replaced_by_new_defaults(tmp_path):
    config = CompletionConfig(hidden_dim=16, output_points=32)
    legacy = CompletionNetwork(config)
    saved = asdict(config)
    del saved["geometry_queries"]
    path = tmp_path / "old.pt"
    torch.save({"model_config": saved, "model": legacy.state_dict()}, path)
    model = VEILNet.from_checkpoint(path)
    assert not model.config.geometry_queries
    assert model.config.bbox_margin == 0.2


def test_incompatible_checkpoint_fails_strictly(tmp_path):
    model = tiny_model()
    state = model.state_dict()
    state.pop(next(iter(state)))
    path = tmp_path / "bad.pt"
    torch.save({"model_config": asdict(model.config), "model": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        VEILNet.from_checkpoint(path)
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match="model_config"):
        VEILNet.from_checkpoint(path)


def test_invalid_export_format_is_rejected(tmp_path):
    tiny_model().save_pretrained(tmp_path)
    config = tmp_path / "config.json"
    data = json.loads(config.read_text())
    data["format_version"] = 999
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported"):
        VEILNet.from_pretrained(tmp_path)


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.ones((4, 4)), np.full((4, 3), np.nan)])
def test_input_validation(tmp_path, points):
    path = tmp_path / "input.npy"
    np.save(path, points)
    with pytest.raises(ValueError):
        read_cloud(path)


def test_sampling_and_archive_keys(tmp_path):
    points = np.arange(48, dtype=np.float32).reshape(16, 3)
    assert prepare_input(points, 16) is points
    np.testing.assert_array_equal(prepare_input(points, 32, 4), prepare_input(points, 32, 4))
    path = tmp_path / "input.npz"
    np.savez(path, partial=points)
    np.testing.assert_array_equal(read_cloud(path), points)
    with pytest.raises(ValueError, match="complete"):
        read_cloud(path, "complete")


def test_hub_loading_is_revision_pinned_and_downloads_only_weights(tmp_path, monkeypatch):
    import huggingface_hub

    tiny_model().save_pretrained(tmp_path)
    calls = []

    def download(filename, **kwargs):
        calls.append((filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    model = VEILNet.from_pretrained("author/VEIL-Net", revision="frozen", local_files_only=True)
    assert isinstance(model, VEILNet)
    assert [name for name, _ in calls] == ["config.json", "model.safetensors"]
    assert all(
        options["revision"] == "frozen" and options["local_files_only"] for _, options in calls
    )
