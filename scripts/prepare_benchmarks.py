"""Build public evidence from frozen local results without rerunning inference."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "candidate": "results/diagnostics/coverage_balance_20260912_150837_6745/comparison.csv",
    "baseline": "results/diagnostics/full_vs_task_snow_cpu/comparison.csv",
    "protocol": "results/diagnostics/coverage_balance_20260912_150837_6745/protocol.json",
    "controls": "results/diagnostics/query_controls/20260916_005454_214586/comparison.csv",
    "dataset_groups": "results/paper_analysis/rapc_coverage_balance_val_20260912_172607_000098/occlusion_all_models.csv",
    "bootstrap": "results/paper_analysis/rapc_coverage_balance_val_20260912_172607_000098/paired_scene_bootstrap.csv",
}
METRICS = [
    "cd_l1",
    "cd_l2",
    "f_score_003",
    "f_score_005",
    "missing_region_cd",
    "observed_region_error",
    "dimension_mae",
]


def read_rows(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "samples", *METRICS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def write_analysis_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_means(rows):
    """Recover dataset means from exhaustive disjoint visibility groups."""
    groups = {}
    for row in rows:
        if (
            row["dataset"] == "ALL"
            or row["model"] not in {"snowflakenet_task_ft", "rapc_coverage_balance"}
            or int(row["samples"]) == 0
        ):
            continue
        groups.setdefault((row["dataset"], row["model"]), []).append(row)
    output = []
    for (dataset, model), parts in sorted(groups.items()):
        count = sum(int(part["samples"]) for part in parts)
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "samples": count,
                **{
                    metric: sum(float(part[metric]) * int(part["samples"]) for part in parts)
                    / count
                    for metric in METRICS
                },
            }
        )
    return output


def main():
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    baseline = next(
        row for row in read_rows(SOURCES["baseline"]) if row["model"] == "snowflakenet_task_ft"
    )
    candidate = read_rows(SOURCES["candidate"])[0]
    write_rows(out / "validation.csv", [baseline, candidate])
    write_rows(out / "ablation.csv", read_rows(SOURCES["controls"]))
    write_analysis_rows(
        out / "dataset_wise.csv", dataset_means(read_rows(SOURCES["dataset_groups"]))
    )
    write_analysis_rows(
        out / "paired_scene_bootstrap.csv",
        [
            row
            for row in read_rows(SOURCES["bootstrap"])
            if row["baseline"] == "snowflakenet_task_ft"
        ],
    )
    source_protocol = json.loads((ROOT / SOURCES["protocol"]).read_text(encoding="utf-8"))
    protocol = {
        key: source_protocol[key]
        for key in ("split", "seed", "eval_points", "missing_threshold", "sample_ids")
    }
    protocol.update(
        {
            "checkpoint_sha256": source_protocol["checkpoints"][0]["sha256"],
            "epoch_index": source_protocol["epoch"],
            "scope": "ID-disjoint validation; aggregate over observations, not objects.",
        }
    )
    (out / "validation_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        key: {
            "path": relative,
            "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
        for key, relative in SOURCES.items()
    }
    (out / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(protocol['sample_ids'])}-observation benchmark evidence in {out}")


if __name__ == "__main__":
    main()
