import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from veil_net import VEILNet, VEILNetConfig
from veil_net.cli import main
from veil_net.evaluation import select_pairs


@pytest.fixture
def prepared(tmp_path):
    torch.set_num_threads(2)
    model = VEILNet(
        VEILNetConfig(
            input_points=16, output_points=32, hidden_dim=16, spatial_tokens=4, spatial_patches=4
        )
    ).eval()
    model.save_pretrained(tmp_path / "model")
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    rng = np.random.default_rng(4)
    for name in ("sample_a", "sample_b"):
        np.savez(
            pairs / f"{name}.npz",
            partial=rng.normal(size=(16, 3)).astype(np.float32),
            complete=rng.normal(size=(32, 3)).astype(np.float32),
        )
    return model, pairs


def test_infer_evaluate_and_report(tmp_path, prepared):
    _, pairs = prepared
    out = tmp_path / "prediction.npz"
    common = ["--model", str(tmp_path / "model")]
    assert (
        main(["infer", *common, "--input", str(pairs / "sample_a.npz"), "--output", str(out)]) == 0
    )
    with np.load(out) as archive:
        assert archive["prediction_complete"].shape == (32, 3)
    report_dir = tmp_path / "evaluation"
    assert (
        main(
            [
                "evaluate",
                *common,
                "--pairs",
                str(pairs),
                "--max-samples",
                "0",
                "--eval-points",
                "16",
                "--output",
                str(report_dir),
            ]
        )
        == 0
    )
    report = json.loads((report_dir / "report.json").read_text())
    assert report["samples"] == 2 and report["eval_points"] == 16
    assert len(report["pair_sha256"]) == 2
    assert report["valid_counts"]["cd_l1"] == 2
    assert report["metrics"]["normal_consistency"] is None
    assert (report_dir / "samples.csv").is_file()


def test_export_has_exact_weights_and_provenance(tmp_path, prepared):
    model, _ = prepared
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "model_config": asdict(model.config), "epoch": 64}, checkpoint
    )
    destination = tmp_path / "hub"
    assert main(["export", "--checkpoint", str(checkpoint), "--output", str(destination)]) == 0
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["epoch_index"] == 64 and provenance["tensor_roundtrip"] == "exact"
    assert len(provenance["source_sha256"]) == 64
    assert "license: mit" in (destination / "README.md").read_text(encoding="utf-8")


def test_cohort_checks_and_seeded_selection(tmp_path, prepared):
    _, pairs = prepared
    with pytest.raises(ValueError, match="Missing"):
        select_pairs(pairs, sample_ids=["absent"])
    with pytest.raises(ValueError, match="unique"):
        select_pairs(pairs, sample_ids=["sample_a", "sample_a"])
    assert select_pairs(pairs, limit=1, seed=4) == select_pairs(pairs, limit=1, seed=4)
    nested = pairs / "duplicate"
    nested.mkdir()
    (nested / "sample_a.npz").touch()
    with pytest.raises(ValueError, match="Duplicate"):
        select_pairs(pairs)


def test_cli_rejects_invalid_sizes(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--model",
                "absent",
                "--pairs",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
                "--max-samples",
                "-1",
            ]
        )
    assert error.value.code == 2


def test_evaluation_uses_protocol_seed(tmp_path, prepared):
    _, pairs = prepared
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"sample_ids": ["sample_a"], "seed": 19, "eval_points": 16}))
    destination = tmp_path / "seeded"
    main(
        [
            "evaluate",
            "--model",
            str(tmp_path / "model"),
            "--pairs",
            str(pairs),
            "--protocol",
            str(protocol),
            "--output",
            str(destination),
        ]
    )
    report = json.loads((destination / "report.json").read_text())
    assert report["seed"] == 19
    assert report["eval_points"] == 16
    assert "weights_sha256" in report["source"]


def test_train_passes_explicit_pair_directory(tmp_path, monkeypatch):
    import rapc_net.training.trainer as trainer

    config = tmp_path / "config.yaml"
    config.write_text("epochs: 1\nbatch_size: 1\noutput_points: 32\n")
    calls = []
    monkeypatch.setattr(
        trainer, "train_completion", lambda **kwargs: calls.append(kwargs) or {"ok": True}
    )
    main(
        [
            "train",
            "--config",
            str(config),
            "--pairs",
            str(tmp_path / "pairs"),
            "--workspace",
            str(tmp_path / "run"),
        ]
    )
    assert calls[0]["pairs_dir"] == (tmp_path / "pairs").resolve()
    assert calls[0]["confirm"] is True
