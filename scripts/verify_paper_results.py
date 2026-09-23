"""Compare a real evaluation report with paper or saved-checkpoint values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("cd_l1", "cd_l2", "f_score_003", "f_score_005", "dimension_mae")


def compare_report(report, expected, protocol, tolerance=0.00001):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    ids = report.get("sample_ids", [])
    checks = {
        "cohort": len(ids) == len(set(ids)) == len(protocol["sample_ids"])
        and set(ids) == set(protocol["sample_ids"]),
        "sample_count": report.get("samples") == len(protocol["sample_ids"]),
        **{
            key: report.get(key) == protocol[key]
            for key in ("seed", "eval_points", "missing_threshold")
        },
    }
    metrics = {}
    for key in METRICS:
        actual = report.get("metrics", {}).get(key)
        target = float(expected[key])
        finite = isinstance(actual, (int, float)) and math.isfinite(actual)
        valid_count = report.get("valid_counts", {}).get(key) == len(protocol["sample_ids"])
        difference = float(actual) - target if finite else None
        metrics[key] = {
            "expected": target,
            "actual": actual if finite else None,
            "difference": difference,
            "matches": finite and valid_count and abs(difference) <= tolerance,
        }
    return {
        "matches": all(checks.values()) and all(row["matches"] for row in metrics.values()),
        "protocol_checks": checks,
        "absolute_tolerance": tolerance,
        "metrics": metrics,
    }


def compare_targets(report, expected, protocol, tolerance=0.00001):
    """Require every target or better; retain all full-cohort protocol checks."""
    result = compare_report(report, expected, protocol, tolerance)
    for key, row in result["metrics"].items():
        row["direction"] = "higher" if key.startswith("f_score") else "lower"
        difference = row["difference"]
        valid_count = report.get("valid_counts", {}).get(key) == len(protocol["sample_ids"])
        row["matches"] = (
            difference is not None
            and valid_count
            and (
                difference >= -tolerance
                if row["direction"] == "higher"
                else difference <= tolerance
            )
        )
    result["matches"] = all(result["protocol_checks"].values()) and all(
        row["matches"] for row in result["metrics"].values()
    )
    result["comparison"] = "all_targets_or_better"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", choices=("paper", "saved"), default="paper")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("reproduce", "target"), default="reproduce")
    args = parser.parse_args(argv)
    filename, model = (
        ("paper_main.csv", "VEIL-Net")
        if args.reference == "paper"
        else ("validation.csv", "rapc_coverage_balance")
    )
    with (ROOT / "benchmarks" / filename).open(encoding="utf-8", newline="") as stream:
        expected = next(row for row in csv.DictReader(stream) if row["model"] == model)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    protocol = json.loads(
        (ROOT / "benchmarks/validation_protocol.json").read_text(encoding="utf-8")
    )
    compare = compare_targets if args.mode == "target" else compare_report
    result = {"reference": args.reference, **compare(report, expected, protocol)}
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
