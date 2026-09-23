"""Train a separate candidate, evaluate unchanged, and check all paper targets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path

import torch
from verify_paper_results import METRICS, compare_targets

from rapc_net.training.trainer import check_training_cohort, train_completion
from rapc_net.utils.io import load_yaml_like
from veil_net.evaluation import evaluate_pairs, select_pairs
from veil_net.model import VEILNet, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_checkpoint(checkpoint, paths, destination, protocol, device):
    identity = sha256_file(checkpoint)
    if destination.exists():
        report_path = destination / "report.json"
        if not report_path.is_file():
            raise ValueError(f"Incomplete evaluation at {destination}; use a new run name")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report["source"].get("checkpoint_sha256") != identity
            or report["sample_ids"] != [p.stem for p in paths]
            or report["device"] != device
            or any(
                report[key] != protocol[key] for key in ("seed", "eval_points", "missing_threshold")
            )
            or report["pair_sha256"] != {p.stem: sha256_file(p) for p in paths}
        ):
            raise ValueError("Existing evaluation does not match this run")
        return report
    model = VEILNet.from_checkpoint(checkpoint, device=device)
    return evaluate_pairs(
        model,
        paths,
        destination,
        device=device,
        eval_points=protocol["eval_points"],
        seed=protocol["seed"],
        missing_threshold=protocol["missing_threshold"],
        provenance={**model.artifact_metadata, "model_config": asdict(model.config)},
    )


def run(args):
    root = args.workspace.resolve()
    config = load_yaml_like(args.config)
    source = root / config["init_checkpoint"]
    if sha256_file(source) != config["init_checkpoint_sha256"]:
        raise ValueError("Warm-start checkpoint hash differs from the verified source")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if state["epoch"] != config["init_checkpoint_epoch"]:
        raise ValueError("Unexpected source epoch")
    for key, value in state["model_config"].items():
        if key in config and config[key] != value:
            raise ValueError(f"Model configuration differs from the source: {key}")
    protocol_path = (root / config["validation_protocol"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    check_training_cohort(state.get("training_sample_ids", []), protocol["sample_ids"])
    train_pairs = (root / args.train_pairs).resolve()
    # Resolve all original training IDs before any subset selection or GPU work.
    select_pairs(train_pairs, sample_ids=state["training_sample_ids"], limit=0)
    paths = select_pairs(
        root / args.val_pairs,
        sample_ids=protocol["sample_ids"],
        limit=args.max_eval_samples,
        seed=protocol["seed"],
    )
    config.update(
        run_name=args.run_name,
        init_checkpoint=str(source.resolve()),
        validation_protocol=str(protocol_path),
        training_sample_ids=state["training_sample_ids"],
        sample_seed=args.seed,
        epochs=state["epoch"] + 1 + args.additional_epochs,
    )
    del state
    destination = root / "results" / "target_finetune" / args.run_name / f"seed_{args.seed}"
    checkpoint = root / "checkpoints" / args.run_name / f"seed_{args.seed}" / "last.pt"
    manifest = {
        "config": config,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "train_pairs": str(train_pairs),
        "eval_sample_ids": [p.stem for p in paths],
        "protocol_sha256": sha256_file(protocol_path),
        "training_device": args.device,
        "evaluation_device": args.eval_device,
        "seed": args.seed,
    }
    if args.resume:
        previous = json.loads((destination / "run.json").read_text(encoding="utf-8"))
        if previous != manifest or not checkpoint.is_file():
            raise ValueError("Resume requires the same arguments and a saved training checkpoint")
        if (destination / "summary.json").exists():
            raise ValueError("Run already completed; choose a new run name")
    else:
        if checkpoint.parent.exists():
            raise FileExistsError(f"Checkpoint directory exists: {checkpoint.parent}")
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination / "run.json", manifest)
        write_json(destination / "training.json", config)
    print(
        f"Baseline evaluation: {len(paths)} samples, {protocol['eval_points']} points", flush=True
    )
    baseline = evaluate_checkpoint(
        source, paths, destination / "baseline", protocol, args.eval_device
    )
    completed = (
        torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"] + 1
        if args.resume
        else config["init_checkpoint_epoch"] + 1
    )
    if completed < config["epochs"]:
        train_completion(
            project_root=root,
            model_name="rapc_net",
            device=args.device,
            seed=args.seed,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            max_samples=args.max_train_samples or None,
            confirm=True,
            resume=args.resume,
            output_points=config["output_points"],
            run_name=args.run_name,
            config_path=destination / "training.json",
            pairs_dir=train_pairs,
        )
    candidate = evaluate_checkpoint(
        checkpoint, paths, destination / "candidate", protocol, args.eval_device
    )
    if baseline["pair_sha256"] != candidate["pair_sha256"]:
        raise ValueError("Validation data changed between the two evaluations")
    with (ROOT / "benchmarks/paper_main.csv").open(encoding="utf-8", newline="") as stream:
        targets = next(row for row in csv.DictReader(stream) if row["model"] == "VEIL-Net")
    verdict = compare_targets(candidate, targets, protocol)
    pilot = bool(args.max_train_samples or args.max_eval_samples)
    summary = {
        "status": "pilot_only"
        if pilot
        else "targets_met"
        if verdict["matches"]
        else "targets_not_met",
        "target_check": verdict,
        "baseline": {key: baseline["metrics"][key] for key in METRICS},
        "candidate": {key: candidate["metrics"][key] for key in METRICS},
        "difference": {
            key: candidate["metrics"][key] - baseline["metrics"][key] for key in METRICS
        },
        "checkpoint": str(checkpoint),
        "scope": "Pilot results are not paper results. All five full-cohort targets must pass.",
        "original_release_modified": False,
    }
    write_json(destination / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0 if pilot or verdict["matches"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/release/veil_metric_finetune.yaml"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-pairs", type=Path, default=Path("cache/completion_pairs/train"))
    parser.add_argument("--val-pairs", type=Path, default=Path("cache/completion_pairs/val"))
    parser.add_argument("--additional-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("run-name must contain only letters, numbers, underscores, and hyphens")
    if (
        args.additional_epochs < 1
        or args.threads < 1
        or min(args.max_train_samples, args.max_eval_samples) < 0
    ):
        parser.error("epochs/threads must be positive; sample limits must be nonnegative")
    torch.set_num_threads(args.threads)
    if (
        any(device.startswith("cuda") for device in (args.device, args.eval_device))
        and not torch.cuda.is_available()
    ):
        parser.error("CUDA is unavailable; specify --device cpu --eval-device cpu")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
