import csv
import importlib.util
import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from veil_net import VEILNet, VEILNetConfig
from veil_net.model import sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "bundle_builder", ROOT / "scripts/build_public_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path, builder, monkeypatch):
    torch.set_num_threads(2)
    root = tmp_path / "source"
    root.mkdir()
    for name in ("benchmarks", "configs/release", "veil_net"):
        (root / name).mkdir(parents=True)
    (root / "LICENSE").write_text("MIT\n")
    (root / "veil_net/model_card.md").write_text(
        "---\nlicense: mit\n---\n# VEIL-Net\n\n## Research Results\n"
    )
    (root / "configs/release/veil_metric_finetune.yaml").write_text("loss: {}\nepochs: 75\n")
    protocol = dict(sample_ids=["a", "b"], seed=0, eval_points=4096, missing_threshold=0.01)
    (root / "benchmarks/validation_protocol.json").write_text(json.dumps(protocol))
    actual = dict(
        cd_l1=0.3862854,
        cd_l2=6.9773651,
        f_score_003=0.6584363,
        f_score_005=0.7542849,
        dimension_mae=0.2284575,
    )
    target = {**actual, "f_score_003": 0.72843, "f_score_005": 0.89253}
    for filename, name, metrics in [
        ("validation.csv", "rapc_coverage_balance", actual),
        ("paper_main.csv", "VEIL-Net", target),
    ]:
        with (root / "benchmarks" / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["model", *actual])
            writer.writeheader()
            writer.writerow(dict(model=name, **metrics))
    model = VEILNet(
        VEILNetConfig(
            input_points=16, output_points=32, hidden_dim=16, spatial_tokens=4, spatial_patches=4
        )
    )
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model.config),
            "epoch": 64,
            "best": 0.1,
            "training_sample_ids": ["train_a"],
        },
        checkpoint,
    )
    export = tmp_path / "export"
    model.save_pretrained(export)
    report = dict(
        **protocol,
        samples=2,
        metrics=actual,
        valid_counts=dict.fromkeys(actual, 2),
        source={"checkpoint_sha256": sha256_file(checkpoint)},
        pair_sha256={"a": "1" * 64, "b": "2" * 64},
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    with (tmp_path / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", *actual])
        writer.writeheader()
        for name in ("a", "b"):
            writer.writerow(dict(sample_id=name, **actual))

    def stage(root, output):
        output.mkdir()
        shutil.copytree(root / "benchmarks", output / "benchmarks")

    monkeypatch.setattr(builder, "stage_release", stage)
    return root, checkpoint, export, report_path, model


def test_single_bundle_has_portable_weights_truthful_status_and_verified_archive(
    builder, source, tmp_path
):
    root, checkpoint, export, report, model = source
    output = tmp_path / "veil-NET"
    result = builder.build_bundle(root, output, checkpoint, export, report)
    assert not result["paper_targets_met"] and result["saved_reference_reproduced"]
    assert builder.verify(output)["integrity_passed"]
    status = json.loads((output / "PUBLICATION_STATUS.json").read_text())
    assert status["actual_metrics"]["f_score_003"] < 0.72843
    recipe = json.loads((output / "configs/release/veil_metric_finetune.yaml").read_text())
    initial = output / recipe["init_checkpoint"]
    assert initial.is_file() and sha256_file(initial) == recipe["init_checkpoint_sha256"]
    state = torch.load(initial, map_location="cpu", weights_only=True)
    assert "optimizer" not in state and state["training_sample_ids"] == ["train_a"]
    restored = VEILNet.from_checkpoint(initial)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    with zipfile.ZipFile(result["archive"]) as archive:
        assert archive.testzip() is None
        assert "veil-NET/model/model.safetensors" in archive.namelist()
        assert "veil-NET/benchmarks/bundled_samples.csv" in archive.namelist()
    with pytest.raises(FileExistsError):
        builder.build_bundle(root, output, checkpoint, export, report)
    (output / "PUBLICATION_STATUS.json").write_text("{}")
    assert not builder.verify(output)["integrity_passed"]
    assert not builder.verify(output)["paper_targets_met"]


def test_target_qualified_bundle_is_blocked_before_writing(builder, source, tmp_path):
    root, checkpoint, export, report, _ = source
    output = tmp_path / "qualified"
    with pytest.raises(ValueError, match="targets are not met"):
        builder.build_bundle(root, output, checkpoint, export, report, require_targets=True)
    assert not output.exists() and not output.with_suffix(".zip").exists()


def test_unrelated_report_and_changed_sample_evidence_are_rejected(builder, source, tmp_path):
    root, checkpoint, export, report_path, _ = source
    report = json.loads(report_path.read_text())
    report["source"] = {"checkpoint_sha256": "wrong"}
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="does not identify"):
        builder.build_bundle(root, tmp_path / "wrong", checkpoint, export, report_path)
    report["source"] = {"checkpoint_sha256": sha256_file(checkpoint)}
    report["metrics"]["f_score_003"] = 0.9
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="reported mean"):
        builder.build_bundle(root, tmp_path / "changed", checkpoint, export, report_path)


def test_integrity_check_detects_modified_weight_file(builder, source, tmp_path):
    root, checkpoint, export, report, _ = source
    output = tmp_path / "bundle"
    builder.build_bundle(root, output, checkpoint, export, report)
    (output / "model/model.safetensors").write_bytes(b"changed")
    result = builder.verify(output)
    assert not result["integrity_passed"]
    assert "SHA-256 mismatch: model/model.safetensors" in result["errors"]
