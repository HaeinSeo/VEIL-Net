"""Bounded multi-recipe training with full-validation checkpoint selection."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import shutil
from pathlib import Path

import torch
from finetune_target_metrics import evaluate_checkpoint, write_json
from verify_paper_results import METRICS, compare_targets

from rapc_net.training.trainer import check_training_cohort, train_completion
from rapc_net.utils.io import load_yaml_like
from veil_net.evaluation import select_pairs
from veil_net.model import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "balanced": {"lambda_eval_fscore": 1.0},
    "coverage": {
        "lambda_eval_fscore": 2.0,
        "lambda_cd": 2.7,
        "lambda_surface": 0.75,
        "lambda_extent": 0.1,
    },
    "geometry": {
        "lambda_eval_fscore": 1.0,
        "lambda_cd": 3.6,
        "lambda_surface": 1.0,
        "lambda_extent": 0.15,
    },
}


def target_rank(report, targets, protocol):
    """Minimize the worst relative target shortfall, then aggregate shortfall."""
    verdict = compare_targets(report, targets, protocol)
    if not all(verdict["protocol_checks"].values()):
        raise ValueError("Checkpoint selection requires the entire frozen validation cohort")
    deficits, signed = [], []
    for key in METRICS:
        actual = report["metrics"].get(key)
        expected = float(targets[key])
        if (
            not isinstance(actual, (float, int))
            or not math.isfinite(actual)
            or actual < 0
            or (key.startswith("f_score") and actual > 1)
            or report["valid_counts"].get(key) != len(protocol["sample_ids"])
        ):
            raise ValueError(f"Invalid full-cohort metric: {key}")
        difference = (expected - actual) if key.startswith("f_score") else (actual - expected)
        signed.append(difference / expected)
        deficits.append(max(0.0, difference - verdict["absolute_tolerance"]) / expected)
    return [max(deficits), sum(deficits), sum(signed)], verdict


def promote_candidate(checkpoint, report, directory, targets, protocol):
    """Keep the closest actual weights, but create winner.pt only on an all-metric pass."""
    rank, verdict = target_rank(report, targets, protocol)
    identity = sha256_file(checkpoint)
    if report.get("source", {}).get("checkpoint_sha256") != identity:
        raise ValueError("Report and candidate checkpoint do not match")
    record_path = directory / "best_candidate.json"
    previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None
    if previous is not None:
        if sha256_file(directory / "best_candidate.pt") != previous["checkpoint_sha256"]:
            raise ValueError("Previously selected checkpoint changed")
    if previous is None or rank < previous["rank"] or verdict["matches"]:
        shutil.copy2(checkpoint, directory / "best_candidate.pt")
        write_json(
            record_path,
            {
                "rank": rank,
                "checkpoint_sha256": identity,
                "source_checkpoint": str(checkpoint),
                "target_check": verdict,
            },
        )
        write_json(directory / "best_candidate_report.json", report)
    if verdict["matches"]:
        shutil.copy2(checkpoint, directory / "winner.pt")
        write_json(directory / "winner_report.json", report)
    return verdict["matches"], rank


def build_trials(profiles, seeds, maximum):
    return [(profile, seed) for seed in seeds for profile in profiles][:maximum]


def evaluate_search_checkpoint(checkpoint, paths, destination, protocol, device):
    # Preserve an interrupted evaluation; retry in a fresh directory without deleting files.
    attempt = destination
    number = 0
    while attempt.exists() and not (attempt / "report.json").is_file():
        number += 1
        attempt = destination.with_name(f"{destination.name}_retry_{number}")
    return evaluate_checkpoint(checkpoint, paths, attempt, protocol, device)


def run(args):
    root = args.workspace.resolve()
    recipe = load_yaml_like(args.config)
    source = (root / recipe["init_checkpoint"]).resolve()
    if sha256_file(source) != recipe["init_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint does not match the verified source hash")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if state["epoch"] != recipe["init_checkpoint_epoch"]:
        raise ValueError("Unexpected initial epoch")
    for key, value in state["model_config"].items():
        if key in recipe and recipe[key] != value:
            raise ValueError(f"Model configuration differs from the source: {key}")
    protocol_path = (root / recipe["validation_protocol"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    check_training_cohort(state["training_sample_ids"], protocol["sample_ids"])
    train_pairs = (root / args.train_pairs).resolve()
    select_pairs(train_pairs, sample_ids=state["training_sample_ids"], limit=0)
    paths = select_pairs(root / args.val_pairs, sample_ids=protocol["sample_ids"], limit=0)
    recipe.update(
        init_checkpoint=str(source),
        training_sample_ids=state["training_sample_ids"],
        validation_protocol=str(protocol_path),
    )
    first_epoch = state["epoch"] + 1
    del state
    targets_path = ROOT / "benchmarks/paper_main.csv"
    with targets_path.open(encoding="utf-8", newline="") as stream:
        targets = next(row for row in csv.DictReader(stream) if row["model"] == "VEIL-Net")
    trials = build_trials(args.profiles, args.seeds, args.max_trials)
    directory = root / "results" / "target_search" / args.run_name
    plan = {
        "recipe": recipe,
        "trials": [list(trial) for trial in trials],
        "profiles": {name: PROFILES[name] for name in args.profiles},
        "epochs_per_trial": args.epochs_per_trial,
        "eval_every": args.eval_every,
        "training_device": args.device,
        "evaluation_device": args.eval_device,
        "threads": args.threads,
        "train_pairs": str(train_pairs),
        "val_pairs": str((root / args.val_pairs).resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "targets_sha256": sha256_file(targets_path),
        "scope": "Hyperparameter and epoch selection on validation, not a held-out test result.",
    }
    if args.plan_only:
        print(
            json.dumps(
                {
                    "output": str(directory),
                    "trials": plan["trials"],
                    "max_additional_epochs": len(trials) * args.epochs_per_trial,
                    "validation_samples": len(paths),
                    "eval_every": args.eval_every,
                },
                indent=2,
            )
        )
        return 0
    if args.resume:
        previous = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        if previous != plan:
            raise ValueError("Resume requires the same training and evaluation plan")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "plan.json", plan)
    if (directory / "winner.pt").exists():
        report = json.loads((directory / "winner_report.json").read_text(encoding="utf-8"))
        if sha256_file(directory / "winner.pt") != report["source"]["checkpoint_sha256"]:
            raise ValueError("Winner checkpoint changed")
        if not target_rank(report, targets, protocol)[1]["matches"]:
            raise ValueError("Saved winner no longer satisfies the targets")
        print(f"Already achieved: {directory / 'winner.pt'}", flush=True)
        return 0
    baseline = evaluate_search_checkpoint(
        source, paths, directory / "baseline", protocol, args.eval_device
    )
    initial_pass, _ = promote_candidate(source, baseline, directory, targets, protocol)
    if initial_pass:
        write_json(
            directory / "summary.json",
            {
                "status": "targets_met",
                "training_required": False,
                "winner": str(directory / "winner.pt"),
                "scope": plan["scope"],
            },
        )
        return 0
    events_path = directory / "evaluations.json"
    events = json.loads(events_path.read_text(encoding="utf-8")) if events_path.exists() else []
    for index, (profile, seed) in enumerate(trials, 1):
        run_name = f"{args.run_name}_{profile}"
        trial_dir = directory / f"{profile}_seed_{seed}"
        checkpoint = root / "checkpoints" / run_name / f"seed_{seed}" / "last.pt"
        if (trial_dir / "completed.json").exists():
            continue
        config = copy.deepcopy(recipe)
        config["loss"].update(PROFILES[profile])
        config.update(
            run_name=run_name, sample_seed=seed, epochs=first_epoch + args.epochs_per_trial
        )
        trial_dir.mkdir(exist_ok=True)
        config_path = trial_dir / "training.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Trial configuration changed")
        else:
            if checkpoint.parent.exists():
                raise FileExistsError(f"Unrelated checkpoint directory exists: {checkpoint.parent}")
            write_json(config_path, config)
        print(
            f"Trial {index}/{len(trials)}: {profile}, seed={seed}, up to {args.epochs_per_trial} additional epochs",
            flush=True,
        )

        def on_epoch_end(saved, epoch):
            additional = epoch + 1 - first_epoch
            if additional % args.eval_every and additional != args.epochs_per_trial:
                return False
            report = evaluate_search_checkpoint(
                saved, paths, trial_dir / f"epoch_{epoch:04d}", protocol, args.eval_device
            )
            if report["pair_sha256"] != baseline["pair_sha256"]:
                raise ValueError("Validation data changed during training")
            passed, rank = promote_candidate(saved, report, directory, targets, protocol)
            event = {
                "profile": profile,
                "seed": seed,
                "epoch": epoch,
                "rank": rank,
                "targets_met": passed,
                "checkpoint_sha256": report["source"]["checkpoint_sha256"],
                "metrics": {key: report["metrics"][key] for key in METRICS},
            }
            if event not in events:
                events.append(event)
                write_json(events_path, events)
            print(json.dumps(event), flush=True)
            return passed

        completed = first_epoch
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            completed = saved["epoch"] + 1
            if not first_epoch < completed <= config["epochs"] or (
                saved.get("loss_config") != config["loss"]
                or set(saved.get("training_sample_ids", [])) != set(config["training_sample_ids"])
                or saved.get("initialization", {}).get("sha256") != recipe["init_checkpoint_sha256"]
            ):
                raise ValueError("Resume checkpoint does not match the planned trial")
            del saved
            if on_epoch_end(checkpoint, completed - 1):
                break
        if completed < config["epochs"]:
            train_completion(
                project_root=root,
                model_name="rapc_net",
                device=args.device,
                seed=seed,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                max_samples=None,
                confirm=True,
                resume=checkpoint.is_file(),
                output_points=config["output_points"],
                run_name=run_name,
                config_path=config_path,
                pairs_dir=train_pairs,
                on_epoch_end=on_epoch_end,
            )
        write_json(
            trial_dir / "completed.json",
            {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)},
        )
        if (directory / "winner.pt").exists():
            break
    passed = (directory / "winner.pt").is_file()
    result = {
        "status": "targets_met" if passed else "budget_exhausted_targets_not_met",
        "evaluations": len(events),
        "best_candidate": str(directory / "best_candidate.pt"),
        "winner": str(directory / "winner.pt") if passed else None,
        "scope": plan["scope"],
    }
    write_json(directory / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/release/veil_metric_finetune.yaml"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-pairs", type=Path, default=Path("cache/completion_pairs/train"))
    parser.add_argument("--val-pairs", type=Path, default=Path("cache/completion_pairs/val"))
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--epochs-per-trial", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("run-name must contain only letters, numbers, underscores, and hyphens")
    if min(args.max_trials, args.epochs_per_trial, args.eval_every, args.threads) < 1:
        parser.error("Training budgets and intervals must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.profiles)) != len(args.profiles):
        parser.error("Seeds and profiles must be unique")
    torch.set_num_threads(args.threads)
    if (
        not args.plan_only
        and any(x.startswith("cuda") for x in (args.device, args.eval_device))
        and not torch.cuda.is_available()
    ):
        parser.error("CUDA is unavailable; use --device cpu --eval-device cpu")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
