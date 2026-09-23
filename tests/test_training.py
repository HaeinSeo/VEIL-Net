from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from veil_net import VEILNet
from veil_net.losses.completion import CompletionLoss, sampled_eval_fscore_loss, soft_fscore_loss
from veil_net.models import CompletionConfig, CompletionNetwork
from veil_net.training.trainer import NpzTorchDataset, check_training_cohort, train_completion


def test_raw_fscore_thresholds_do_not_follow_relative_normalization():
    torch.set_num_threads(2)
    target = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    pred = (target + torch.tensor([0.0, 0.0, 0.04])).requires_grad_()
    criterion = CompletionLoss(
        lambda_cd=0,
        lambda_preserve=0,
        lambda_risk=0,
        lambda_uncertainty=0,
        lambda_uniform=0,
        lambda_balance=0,
        lambda_eval_fscore=0.5,
        relative_geometry_loss=True,
    )
    loss, parts = criterion({"completed_points": pred}, {"complete": target})
    expected = soft_fscore_loss(pred, target, thresholds=(0.03, 0.05))
    torch.testing.assert_close(loss, expected * 0.5)
    assert parts["raw_eval_fscore"] == pytest.approx(expected.item())
    gradient = torch.autograd.grad(loss, pred)[0]
    assert torch.isfinite(gradient).all() and gradient[:, :, 2].min() > 0
    assert sampled_eval_fscore_loss(pred - gradient * 0.001, target) < expected


@pytest.mark.parametrize("points", [0, -1, 1.5, True])
def test_invalid_fscore_resolution_is_rejected(points):
    with pytest.raises(ValueError):
        CompletionLoss(eval_fscore_points=points)


def test_training_cohort_is_disjoint_and_complete(tmp_path):
    check_training_cohort(["train"], ["val"])
    for ids in (["val"], ["train", "train"], []):
        with pytest.raises(ValueError):
            check_training_cohort(ids, ["val"])
    (tmp_path / "a.npz").touch()
    (tmp_path / "b.npz").touch()
    assert [p.stem for p in NpzTorchDataset(tmp_path, sample_ids=["b"]).paths] == ["b"]
    with pytest.raises(ValueError, match="Missing"):
        NpzTorchDataset(tmp_path, sample_ids=["c"], max_samples=1)


def test_required_training_manifest_cannot_be_skipped(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("require_training_manifest: true\n")
    with pytest.raises(ValueError, match="requires a training manifest"):
        train_completion(
            project_root=tmp_path,
            model_name="veil_net",
            device="cpu",
            seed=0,
            epochs=1,
            batch_size=1,
            max_samples=None,
            confirm=True,
            resume=False,
            config_path=config,
        )


def test_training_checkpoint_can_be_loaded_and_resumed(tmp_path, monkeypatch):
    import veil_net.training.trainer as trainer

    torch.set_num_threads(2)
    config = CompletionConfig(input_points=8, output_points=16, hidden_dim=16)
    monkeypatch.setattr(trainer, "create_model", lambda *args, **kwargs: CompletionNetwork(config))
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    rng = np.random.default_rng(0)
    np.savez(
        pairs / "sample.npz",
        partial=rng.normal(size=(8, 3)).astype(np.float32),
        complete=rng.normal(size=(16, 3)).astype(np.float32),
    )
    args = dict(
        project_root=tmp_path,
        model_name="veil_net",
        device="cpu",
        seed=0,
        batch_size=1,
        max_samples=None,
        confirm=True,
        output_points=16,
        pairs_dir=pairs,
        run_name="tiny",
    )
    result = train_completion(**args, epochs=1, resume=False)
    checkpoint = Path(result["checkpoint"])
    restored = VEILNet.from_checkpoint(checkpoint)
    assert asdict(restored.config) == asdict(config)
    assert result["epochs_completed"] == 1
    result = train_completion(**args, epochs=2, resume=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert result["epochs_completed"] == 1
    assert state["epoch"] == 1 and state["training_sample_ids"] == ["sample"]
