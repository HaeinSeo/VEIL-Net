import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(name):
    with (ROOT / "benchmarks" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_all_paper_tables_are_rendered_in_public_documents():
    renderer = load_script("render_paper_results")
    for filename, korean in [
        ("docs/RESULTS.md", False),
        ("README_KO.md", True),
        ("benchmarks/README.md", False),
        ("veil_net/model_card.md", False),
    ]:
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert renderer.render(ROOT, korean=korean) in text
    assert len(records("paper_main.csv")) == 6
    assert len(records("paper_improvements.csv")) == 5
    assert len(records("paper_coordinate_ablation.csv")) == 2
    assert len(records("paper_dataset_wise.csv")) == 6
    assert len(records("paper_bootstrap.csv")) == 4


def test_dataset_and_bootstrap_rows_match_saved_analyses():
    datasets = {"T-LESS": "tless", "TUD-L": "tudl", "YCB-Video": "ycb_video"}
    models = {"SnowflakeNet FT": "snowflakenet_task_ft", "VEIL-Net": "rapc_coverage_balance"}
    raw = {(r["dataset"], r["model"]): r for r in records("dataset_wise.csv")}
    for row in records("paper_dataset_wise.csv"):
        source = raw[datasets[row["dataset"]], models[row["model"]]]
        assert row["samples"] == source["samples"]
        for metric in ("cd_l1", "f_score_003", "missing_region_cd", "dimension_mae"):
            assert float(row[metric]) == pytest.approx(float(source[metric]), abs=0.000005)
    raw = {r["metric"]: r for r in records("paired_scene_bootstrap.csv")}
    for row in records("paper_bootstrap.csv"):
        for key, source_key in [
            ("mean_difference", "ours_minus_baseline"),
            ("ci95_low", "ci95_low"),
            ("ci95_high", "ci95_high"),
        ]:
            assert float(row[key]) == pytest.approx(
                float(raw[row["metric"]][source_key]), abs=0.000005
            )


def test_paper_improvement_units_and_values():
    main = {r["model"]: r for r in records("paper_main.csv")}
    for row in records("paper_improvements.csv"):
        ours = float(main["VEIL-Net"][row["metric"]])
        baseline = float(main["SnowflakeNet FT"][row["metric"]])
        expected = (
            (ours - baseline) * 100
            if row["unit"] == "percentage_points"
            else (1 - ours / baseline) * 100
        )
        assert float(row["improvement"]) == pytest.approx(expected, abs=0.005)


def test_reproduction_check_rejects_f_score_mismatch_and_wrong_cohort():
    verifier = load_script("verify_paper_results")
    protocol = json.loads((ROOT / "benchmarks/validation_protocol.json").read_text())
    saved = next(r for r in records("validation.csv") if r["model"] == "rapc_coverage_balance")
    paper = next(r for r in records("paper_main.csv") if r["model"] == "VEIL-Net")
    report = {
        **protocol,
        "samples": 929,
        "metrics": {key: float(saved[key]) for key in verifier.METRICS},
        "valid_counts": dict.fromkeys(verifier.METRICS, 929),
    }
    assert verifier.compare_report(report, saved, protocol)["matches"]
    result = verifier.compare_report(report, paper, protocol)
    assert not result["matches"]
    assert result["metrics"]["cd_l1"]["matches"]
    assert not result["metrics"]["f_score_003"]["matches"]
    assert not result["metrics"]["f_score_005"]["matches"]
    report["sample_ids"] = report["sample_ids"][:-1]
    assert not verifier.compare_report(report, saved, protocol)["matches"]
