import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from veil_net.model import sha256_file

ROOT = Path(__file__).resolve().parents[1]
TARGETS = dict(
    cd_l1=0.38629, cd_l2=6.97737, f_score_003=0.72843, f_score_005=0.89253, dimension_mae=0.22846
)
PROTOCOL = dict(sample_ids=["val_a", "val_b"], seed=0, eval_points=4096, missing_threshold=0.01)


@pytest.fixture
def search(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "target_search", ROOT / "scripts/search_target_metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(metrics, checkpoint):
    return dict(
        **PROTOCOL,
        samples=2,
        metrics=metrics,
        valid_counts=dict.fromkeys(TARGETS, 2),
        source={"checkpoint_sha256": sha256_file(checkpoint)},
        pair_sha256={"val_a": "a", "val_b": "b"},
    )


def test_target_selection_requires_all_metrics_and_matching_weights(search, tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"real artifact identity")
    below = {**TARGETS, "f_score_005": 0.8}
    passed, rank = search.promote_candidate(
        checkpoint, report(below, checkpoint), tmp_path, TARGETS, PROTOCOL
    )
    assert not passed and rank[0] > 0
    assert (tmp_path / "best_candidate.pt").is_file()
    assert not (tmp_path / "winner.pt").exists()
    passed, _ = search.promote_candidate(
        checkpoint, report(TARGETS, checkpoint), tmp_path, TARGETS, PROTOCOL
    )
    assert passed and (tmp_path / "winner.pt").read_bytes() == checkpoint.read_bytes()
    forged = report(TARGETS, checkpoint)
    forged["source"]["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="do not match"):
        search.promote_candidate(checkpoint, forged, tmp_path, TARGETS, PROTOCOL)


def test_ranking_rejects_partial_cohort_and_invalid_values(search, tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.touch()
    partial = report(TARGETS, checkpoint)
    partial["sample_ids"] = ["val_a"]
    with pytest.raises(ValueError, match="entire"):
        search.target_rank(partial, TARGETS, PROTOCOL)
    for value in (float("nan"), -0.1, 1.01):
        with pytest.raises(ValueError, match="Invalid"):
            search.target_rank(
                report({**TARGETS, "f_score_003": value}, checkpoint), TARGETS, PROTOCOL
            )


def test_budget_interleaves_profiles_and_seeds(search):
    assert search.build_trials(["a", "b"], [0, 1], 3) == [("a", 0), ("b", 0), ("a", 1)]


def test_incomplete_evaluation_is_preserved(search, tmp_path, monkeypatch):
    incomplete = tmp_path / "epoch_0069"
    incomplete.mkdir()
    calls = []
    monkeypatch.setattr(search, "evaluate_checkpoint", lambda *args: calls.append(args))
    search.evaluate_search_checkpoint("model.pt", [], incomplete, PROTOCOL, "cpu")
    assert calls[0][2].name == "epoch_0069_retry_1"
    assert incomplete.is_dir()


@pytest.mark.parametrize("achieved", [True, False])
def test_search_stops_on_success_or_budget_without_false_winner(
    search, tmp_path, monkeypatch, achieved
):
    source = tmp_path / "initial.pt"
    torch.save({"epoch": 64, "model_config": {}, "training_sample_ids": ["train_a"]}, source)
    for split, names in [("train", ["train_a"]), ("val", PROTOCOL["sample_ids"])]:
        folder = tmp_path / split
        folder.mkdir()
        for name in names:
            (folder / f"{name}.npz").touch()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(PROTOCOL))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            dict(
                init_checkpoint=str(source),
                init_checkpoint_sha256=sha256_file(source),
                init_checkpoint_epoch=64,
                validation_protocol=str(protocol_path),
                loss={},
                batch_size=1,
                output_points=32,
            )
        )
    )
    args = argparse.Namespace(
        workspace=tmp_path,
        config=config,
        train_pairs=Path("train"),
        val_pairs=Path("val"),
        profiles=["balanced", "coverage"],
        seeds=[0],
        max_trials=2,
        run_name="search",
        epochs_per_trial=2,
        eval_every=1,
        device="cpu",
        eval_device="cpu",
        threads=2,
        plan_only=False,
        resume=False,
    )
    calls = []

    def fake_evaluation(checkpoint, *args):
        metrics = TARGETS if checkpoint != source and achieved else {**TARGETS, "f_score_003": 0.6}
        return report(metrics, checkpoint)

    def fake_train(**kwargs):
        calls.append(kwargs)
        checkpoint = tmp_path / "checkpoints" / kwargs["run_name"] / "seed_0" / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(kwargs["run_name"].encode())
        stopped = kwargs["on_epoch_end"](checkpoint, 66)
        assert stopped == achieved

    monkeypatch.setattr(search, "evaluate_search_checkpoint", fake_evaluation)
    monkeypatch.setattr(search, "train_completion", fake_train)
    assert search.run(args) == (0 if achieved else 2)
    output = tmp_path / "results/target_search/search"
    assert (output / "winner.pt").exists() == achieved
    assert len(calls) == (1 if achieved else 2)
    args.resume = True
    assert search.run(args) == (0 if achieved else 2)
    assert len(calls) == (1 if achieved else 2)


def test_real_trainer_hook_stops_after_saved_epoch_and_preserves_rng(tmp_path, monkeypatch):
    import rapc_net.training.trainer as trainer
    from rapc_net.models import RAPCNet, RAPCNetConfig

    torch.set_num_threads(2)
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    points = np.random.default_rng(1).normal(size=(16, 3)).astype(np.float32)
    np.savez(pairs / "train.npz", partial=points, complete=points)
    monkeypatch.setattr(
        trainer,
        "create_model",
        lambda *args, **kwargs: RAPCNet(
            RAPCNetConfig(input_points=16, output_points=32, hidden_dim=16)
        ),
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"amp": False, "loss": {"lambda_uniform": 0}}))
    calls = []

    def callback(path, epoch):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["epoch"] == epoch == 0
        calls.append(torch.rand(3))
        return True

    result = trainer.train_completion(
        project_root=tmp_path,
        model_name="rapc_net",
        device="cpu",
        seed=0,
        epochs=3,
        batch_size=1,
        max_samples=None,
        confirm=True,
        resume=False,
        output_points=32,
        config_path=config,
        pairs_dir=pairs,
        on_epoch_end=callback,
    )
    assert result["epochs_completed"] == len(calls) == 1
    torch.testing.assert_close(torch.rand(3), calls[0], rtol=0, atol=0)
