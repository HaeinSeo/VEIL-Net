"""Build one portable source + measured-weights ZIP, with honest result provenance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch
from prepare_release import stage_release
from verify_bundle import digest, verify
from verify_paper_results import METRICS, compare_report, compare_targets

from rapc_net.training.trainer import check_training_cohort
from rapc_net.utils.io import load_yaml_like
from veil_net.model import VEILNet

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reference(root, name, model):
    with (root / "benchmarks" / name).open(encoding="utf-8", newline="") as stream:
        return next(row for row in csv.DictReader(stream) if row["model"] == model)


def verify_report_identity(report, checkpoint_sha, model_metadata):
    source = report.get("source", {})
    raw_identity = source.get("checkpoint_sha256") == checkpoint_sha
    portable_identity = all(
        source.get(key) == model_metadata[key] for key in ("weights_sha256", "config_sha256")
    )
    if not (raw_identity or portable_identity):
        raise ValueError("Evaluation report does not identify the bundled weights")
    ids = report.get("sample_ids", [])
    if set(report.get("pair_sha256", {})) != set(ids):
        raise ValueError("Evaluation report is missing pair identities")
    for key in METRICS:
        value = report.get("metrics", {}).get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid measured metric: {key}")
        if key.startswith("f_score") and value > 1:
            raise ValueError(f"Invalid F-score: {key}")


def verify_sample_evidence(report, samples_path):
    with samples_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(report["sample_ids"]):
        raise ValueError("Per-sample evidence does not match the report cohort")
    for key in METRICS:
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(value) for value in values) or not math.isclose(
            math.fsum(values) / len(values),
            report["metrics"][key],
            abs_tol=1e-10,
            rel_tol=1e-10,
        ):
            raise ValueError(f"Per-sample evidence does not match the reported mean: {key}")


def build_bundle(root, output, checkpoint, model_dir, report_path, require_targets=False):
    root, output = root.resolve(), output.resolve()
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise FileExistsError("Choose a new bundle directory and ZIP name")
    protocol = json.loads(
        (root / "benchmarks/validation_protocol.json").read_text(encoding="utf-8")
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    original_sha = digest(checkpoint)
    model = VEILNet.from_checkpoint(checkpoint)
    exported = VEILNet.from_pretrained(model_dir)
    if asdict(model.config) != asdict(exported.config) or model.config.oracle_risk:
        raise ValueError("Model configurations differ or depend on oracle inputs")
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(exported.state_dict()[key], tensor, rtol=0, atol=0)
    verify_report_identity(report, original_sha, exported.artifact_metadata)
    samples_path = report_path.with_name("samples.csv")
    verify_sample_evidence(report, samples_path)
    targets = compare_targets(report, reference(root, "paper_main.csv", "VEIL-Net"), protocol)
    saved = compare_report(
        report, reference(root, "validation.csv", "rapc_coverage_balance"), protocol
    )
    if not all(targets["protocol_checks"].values()) or any(
        report["valid_counts"].get(key) != len(protocol["sample_ids"]) for key in METRICS
    ):
        raise ValueError("A complete frozen-protocol evaluation is required")
    if require_targets and not targets["matches"]:
        raise ValueError("Paper targets are not met; refusing a target-qualified release")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    check_training_cohort(state["training_sample_ids"], protocol["sample_ids"])
    stage_release(root, output)
    model.save_pretrained(output / "model")
    shutil.copy2(root / "LICENSE", output / "model/LICENSE")
    metadata = {
        "source_checkpoint_sha256": original_sha,
        "weights_sha256": digest(output / "model/model.safetensors"),
        "config_sha256": digest(output / "model/config.json"),
        "epoch_index": int(state["epoch"]),
        "tensor_roundtrip": "exact",
    }
    write_json(output / "model/provenance.json", metadata)
    # The new report keeps its numerical results and data hashes; only portable weight
    # identity is added after exact tensor equality with the evaluated source is checked.
    linked_report = {
        **report,
        "source": {key: metadata[key] for key in ("weights_sha256", "config_sha256")},
        "original_report_sha256": digest(report_path),
        "original_source": {
            key: value for key, value in report.get("source", {}).items() if key.endswith("sha256")
        },
    }
    write_json(output / "benchmarks/bundled_evaluation.json", linked_report)
    shutil.copy2(samples_path, output / "benchmarks/bundled_samples.csv")
    write_json(output / "model/evaluation.json", linked_report)
    status = {
        "status": "paper_targets_met"
        if targets["matches"]
        else "measured_reference_below_paper_targets",
        "paper_targets_met": targets["matches"],
        "saved_reference_reproduced": saved["matches"],
        "paper_target_check": targets,
        "actual_metrics": {key: report["metrics"][key] for key in METRICS},
        "weights": metadata,
        "data_included": False,
        "scope": "Code and measured reference weights are reusable; unachieved scores are not claimed.",
    }
    write_json(output / "PUBLICATION_STATUS.json", status)
    card = (root / "veil_net/model_card.md").read_text(encoding="utf-8")
    measured = "## These Bundled Weights\n\n" + (
        "All five paper targets passed.\n\n"
        if targets["matches"]
        else "These weights have not met both externally reported F-score targets.\n\n"
    )
    measured += "| Metric | Measured |\n| --- | ---: |\n"
    measured += "".join(f"| {key} | {report['metrics'][key]:.8f} |\n" for key in METRICS)
    measured += "\nFull evidence: [evaluation.json](evaluation.json). Historical research tables below are separately attributed.\n\n"
    card = card.replace("## Research Results", measured + "## Research Results", 1)
    (output / "model/README.md").write_text(card, encoding="utf-8")
    weights = output / "weights"
    weights.mkdir()
    initial = weights / "initial.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model.config),
            "epoch": int(state["epoch"]),
            "best": float(state["best"]),
            "training_sample_ids": state["training_sample_ids"],
            "loss_config": state.get("loss_config", {}),
        },
        initial,
    )
    restored = VEILNet.from_checkpoint(initial)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor, rtol=0, atol=0)
    recipe = load_yaml_like(root / "configs/release/veil_metric_finetune.yaml")
    recipe.update(
        init_checkpoint="weights/initial.pt",
        init_checkpoint_sha256=digest(initial),
        init_checkpoint_epoch=int(state["epoch"]),
        epochs=int(state["epoch"]) + 11,
    )
    write_json(output / "configs/release/veil_metric_finetune.yaml", recipe)
    write_json(
        weights / "provenance.json",
        {
            "original_checkpoint_sha256": original_sha,
            "initial_checkpoint_sha256": digest(initial),
            "optimizer_included": False,
            "training_samples": len(state["training_sample_ids"]),
            "purpose": "Exact-weight warm start with a fresh optimizer, not an optimizer-state resume.",
        },
    )
    # Store frozen evaluation records beside the code, never cache/ or source datasets.
    files = sorted(
        p for p in output.rglob("*") if p.is_file() and p.name != "release_manifest.json"
    )
    write_json(
        output / "release_manifest.json",
        {
            "files": len(files),
            "sha256": {p.relative_to(output).as_posix(): digest(p) for p in files},
        },
    )
    if not verify(output)["integrity_passed"]:
        raise RuntimeError("Bundle integrity verification failed")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in [*files, output / "release_manifest.json"]:
            stream.write(path, (Path(output.name) / path.relative_to(output)).as_posix())
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise RuntimeError("ZIP verification failed")
    return {
        "archive": str(archive),
        "files": len(files),
        "paper_targets_met": targets["matches"],
        "saved_reference_reproduced": saved["matches"],
        "archive_sha256": digest(archive),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "releases/veil-NET")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-targets", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    print(
        json.dumps(
            build_bundle(
                ROOT, args.output, args.checkpoint, args.model, args.report, args.require_targets
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
