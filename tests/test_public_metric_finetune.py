import importlib.util
from pathlib import Path

import pytest
import torch

from rapc_net.losses.completion import CompletionLoss, sampled_eval_fscore_loss, soft_fscore_loss
from rapc_net.training.trainer import NpzTorchDataset, check_training_cohort

ROOT = Path(__file__).resolve().parents[1]


def criterion(**kwargs):
    return CompletionLoss(
        lambda_cd=0,
        lambda_preserve=0,
        lambda_risk=0,
        lambda_uncertainty=0,
        lambda_uniform=0,
        lambda_balance=0,
        lambda_eval_fscore=0.5,
        **kwargs,
    )


def test_raw_threshold_loss_is_independent_of_relative_frame():
    torch.set_num_threads(2)
    target = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    pred = (target + torch.tensor([0.0, 0.0, 0.04])).requires_grad_()
    loss, parts = criterion(relative_geometry_loss=True)(
        {"completed_points": pred}, {"complete": target}
    )
    expected = soft_fscore_loss(pred, target, thresholds=(0.03, 0.05))
    torch.testing.assert_close(loss, expected * 0.5)
    assert parts["raw_eval_fscore"] == pytest.approx(expected.item())
    gradient = torch.autograd.grad(loss, pred)[0]
    assert torch.isfinite(gradient).all() and gradient[:, :, 2].min() > 0
    assert sampled_eval_fscore_loss(pred - gradient * 0.001, target) < expected


def test_thinning_is_seeded_and_uses_independent_samples():
    torch.manual_seed(1)
    target = torch.randn(1, 16, 3) * 0.03
    pred = torch.randn(1, 18, 3, requires_grad=True) * 0.03
    torch.manual_seed(12)
    actual = sampled_eval_fscore_loss(pred, target, 8)
    torch.manual_seed(12)
    p = torch.randperm(18)[:8]
    q = torch.randperm(16)[:8]
    expected = soft_fscore_loss(pred[:, p], target[:, q], thresholds=(0.03, 0.05))
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, pred)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[:, sorted(set(range(18)) - set(p.tolist()))].abs().sum() == 0


@pytest.mark.parametrize("points", [0, -1, 1.5, True])
def test_invalid_fscore_resolution_is_rejected(points):
    with pytest.raises(ValueError):
        criterion(eval_fscore_points=points)


def test_training_cohort_must_be_disjoint_and_complete(tmp_path):
    check_training_cohort(["train"], ["val"])
    for train_ids in (["val"], ["train", "train"], []):
        with pytest.raises(ValueError):
            check_training_cohort(train_ids, ["val"])
    (tmp_path / "a.npz").touch()
    (tmp_path / "b.npz").touch()
    data = NpzTorchDataset(tmp_path, sample_ids=["b"])
    assert [p.stem for p in data.paths] == ["b"]
    with pytest.raises(ValueError, match="Missing"):
        NpzTorchDataset(tmp_path, sample_ids=["c"], max_samples=1)


def test_target_recipe_cannot_skip_training_manifest(tmp_path):
    from rapc_net.training.trainer import train_completion

    config = tmp_path / "recipe.yaml"
    config.write_text("require_training_manifest: true\n")
    with pytest.raises(ValueError, match="requires a training manifest"):
        train_completion(
            project_root=tmp_path,
            model_name="rapc_net",
            device="cpu",
            seed=0,
            epochs=1,
            batch_size=1,
            max_samples=None,
            confirm=True,
            resume=False,
            config_path=config,
        )


def test_target_gate_requires_all_five_and_full_cohort():
    spec = importlib.util.spec_from_file_location(
        "verifier", ROOT / "scripts/verify_paper_results.py"
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    protocol = dict(sample_ids=["a", "b"], seed=0, eval_points=4096, missing_threshold=0.01)
    targets = dict(
        cd_l1=0.38629,
        cd_l2=6.97737,
        f_score_003=0.72843,
        f_score_005=0.89253,
        dimension_mae=0.22846,
    )
    report = dict(
        **protocol, samples=2, metrics=targets.copy(), valid_counts=dict.fromkeys(targets, 2)
    )
    assert verifier.compare_targets(report, targets, protocol)["matches"]
    report["metrics"]["cd_l1"] = 0.3
    report["metrics"]["f_score_005"] = 0.95
    assert verifier.compare_targets(report, targets, protocol)["matches"]
    report["metrics"]["f_score_003"] = 0.7
    assert not verifier.compare_targets(report, targets, protocol)["matches"]
    report["metrics"] = targets.copy()
    report["sample_ids"] = ["a"]
    assert not verifier.compare_targets(report, targets, protocol)["matches"]


def test_resumed_evaluation_rejects_changed_data(tmp_path, monkeypatch):
    import json

    from veil_net.model import sha256_file

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "runner", ROOT / "scripts/finetune_target_metrics.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"identity only")
    pair = tmp_path / "a.npz"
    pair.write_bytes(b"original")
    destination = tmp_path / "evaluation"
    destination.mkdir()
    protocol = dict(seed=0, eval_points=4096, missing_threshold=0.01)
    report = dict(
        **protocol,
        source={"checkpoint_sha256": sha256_file(checkpoint)},
        sample_ids=["a"],
        device="cpu",
        pair_sha256={"a": sha256_file(pair)},
    )
    (destination / "report.json").write_text(json.dumps(report))
    assert runner.evaluate_checkpoint(checkpoint, [pair], destination, protocol, "cpu") == report
    pair.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        runner.evaluate_checkpoint(checkpoint, [pair], destination, protocol, "cpu")
