"""Bounded evaluation with explicit cohorts, sampling, and per-sample results."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np

from rapc_net.evaluation.metrics import evaluate_completion_torch
from veil_net.inference import predict, read_cloud
from veil_net.model import VEILNet, sha256_file


def select_pairs(
    directory: Path, *, sample_ids: list[str] | None = None, limit: int = 16, seed: int = 0
) -> list[Path]:
    if limit < 0:
        raise ValueError("max-samples must be zero (all) or positive")
    indexed = {}
    for path in sorted(directory.rglob("*.npz")):
        if path.stem in indexed:
            raise ValueError(f"Duplicate sample ID: {path.stem}")
        indexed[path.stem] = path
    if sample_ids is not None:
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Protocol sample_ids must be nonempty and unique")
        missing = set(sample_ids) - indexed.keys()
        if missing:
            raise ValueError(f"Missing {len(missing)} protocol pairs, e.g. {sorted(missing)[0]}")
        indexed = {key: indexed[key] for key in sample_ids}
    ids = sorted(indexed)
    if not ids:
        raise FileNotFoundError(f"No completion pairs found in {directory}")
    if limit and limit < len(ids):
        ids = sorted(random.Random(seed).sample(ids, limit))
    return [indexed[key] for key in ids]


def evaluate_pairs(
    model: VEILNet,
    paths: list[Path],
    output: Path,
    *,
    device: str = "cpu",
    eval_points: int = 1024,
    seed: int = 0,
    missing_threshold: float = 0.01,
    provenance: dict | None = None,
) -> dict:
    if not paths or eval_points < 1 or missing_threshold <= 0:
        raise ValueError("Evaluation requires samples, positive resolution, and positive threshold")
    if output.exists():
        raise FileExistsError(f"Choose a new evaluation directory: {output}")
    output.mkdir(parents=True)
    rows = []
    pair_hashes = {}
    for index, path in enumerate(paths, 1):
        partial, target = read_cloud(path), read_cloud(path, "complete")
        if len(partial) != model.config.input_points:
            raise ValueError(
                f"{path.name}: expected {model.config.input_points} prepared input points"
            )
        result = predict(model, partial, seed=seed)
        metrics = evaluate_completion_torch(
            result["prediction_complete"],
            target,
            partial,
            f_threshold=missing_threshold,
            device=device,
            chunk=512,
            max_points=eval_points,
            seed=seed,
        ).as_dict()
        for key, value in metrics.items():
            if not math.isfinite(value) and key not in {"missing_region_cd", "normal_consistency"}:
                raise FloatingPointError(f"Nonfinite {key} for {path.stem}")
        rows.append({"sample_id": path.stem, **metrics})
        pair_hashes[path.stem] = sha256_file(path)
        print(f"Evaluated {index}/{len(paths)}: {path.stem}", flush=True)
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (v if not isinstance(v, float) or math.isfinite(v) else "")
                    for k, v in row.items()
                }
            )
    means, valid_counts = {}, {}
    for key in rows[0]:
        if key == "sample_id":
            continue
        values = [row[key] for row in rows if math.isfinite(row[key])]
        means[key] = float(np.mean(values)) if values else None
        valid_counts[key] = len(values)
    report = {
        "samples": len(paths),
        "metrics": means,
        "valid_counts": valid_counts,
        "sample_ids": [path.stem for path in paths],
        "pair_sha256": pair_hashes,
        "seed": seed,
        "eval_points": eval_points,
        "missing_threshold": missing_threshold,
        "device": device,
        "source": provenance or {},
        "scope": "Metrics for the listed cohort and evaluation resolution only.",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report
