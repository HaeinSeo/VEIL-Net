"""Render all manuscript tables consistently in public documentation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- BEGIN PAPER RESULTS -->"
END = "<!-- END PAPER RESULTS -->"
LABELS = {
    "cd_l1": "CD-L1",
    "cd_l2": "CD-L2",
    "f_score_003": "F@0.03",
    "f_score_005": "F@0.05",
    "dimension_mae": "Dimension MAE",
}


def rows(root, name):
    with (root / "benchmarks" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def table(headers, values):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in values),
        ]
    )


def render(root, korean=False):
    sections = []
    if korean:
        sections.append(
            "첨부 원고의 전체 결과이다. 주 비교표의 VEIL-Net은 **외부 실행 보고값**이며, "
            "좌표 조건화는 **50-epoch 제거 실험**, 데이터셋별 결과와 bootstrap은 "
            "**저장 체크포인트**에 해당한다."
        )
    else:
        sections.append(
            "Complete results from the author-supplied manuscript. The main VEIL-Net row is "
            "an **author-reported external run**; coordinate conditioning uses **50-epoch "
            "controls**; dataset-wise results and bootstrap intervals describe the "
            "**saved checkpoint**."
        )
    sections.append("### 평가 데이터" if korean else "### Evaluation Cohort")
    sections.append(
        table(
            ["Dataset", "Samples"],
            [[r["dataset"], r["samples"]] for r in rows(root, "dataset_counts.csv")],
        )
    )
    sections.append("### 전체 모델 비교" if korean else "### Overall Completion")
    metrics = ["cd_l1", "cd_l2", "f_score_003", "f_score_005", "dimension_mae"]
    values = []
    for row in rows(root, "paper_main.csv"):
        values.append([row["model"], *[row[key] for key in metrics]])
        if row["model"] == "VEIL-Net":
            values[-1][0] = "VEIL-Net (external)"
            values[-1] = [f"**{value}**" for value in values[-1]]
    sections.append(
        table(["Method", "CD-L1 ↓", "CD-L2 ↓", "F@0.03 ↑", "F@0.05 ↑", "Dim. MAE ↓"], values)
    )
    sections.append(
        "### SnowflakeNet FT 대비 개선" if korean else "### Improvement over SnowflakeNet FT"
    )
    values = []
    for row in rows(root, "paper_improvements.csv"):
        suffix = "%p" if row["unit"] == "percentage_points" else "%"
        direction = "↑" if row["direction"] == "increase" else "↓"
        values.append([LABELS[row["metric"]], f"**{direction} {row['improvement']}{suffix}**"])
    sections.append(table(["Metric", "Improvement"], values))
    sections.append(
        "### 좌표 조건화 제거 실험" if korean else "### Coordinate-Conditioning Ablation"
    )
    sections.append(
        table(
            ["Setting", "CD-L2 ↓", "F@0.03 ↑", "F@0.05 ↑", "Missing CD ↓", "Observed Error ↓"],
            [
                [
                    r["setting"],
                    r["cd_l2"],
                    r["f_score_003"],
                    r["f_score_005"],
                    r["missing_region_cd"],
                    r["observed_region_error"],
                ]
                for r in rows(root, "paper_coordinate_ablation.csv")
            ],
        )
    )
    sections.append(
        "좌표 조건화로 **Missing CD 8.15% 감소**, **CD-L2 5.85% 감소**, "
        "**F@0.03 2.11%p 증가**가 나타났다."
        if korean
        else "Coordinate conditioning reduces **Missing CD by 8.15%** and **CD-L2 by 5.85%**, "
        "and increases **F@0.03 by 2.11 percentage points** against its removal control."
    )
    sections.append("### 데이터셋별 분석" if korean else "### Dataset-Wise Analysis")
    sections.append(
        table(
            ["Dataset", "Samples", "Method", "CD-L1 ↓", "F@0.03 ↑", "Missing CD ↓", "Dim. MAE ↓"],
            [
                [
                    r["dataset"],
                    r["samples"],
                    r["model"],
                    r["cd_l1"],
                    r["f_score_003"],
                    r["missing_region_cd"],
                    r["dimension_mae"],
                ]
                for r in rows(root, "paper_dataset_wise.csv")
            ],
        )
    )
    sections.append(
        "세 데이터셋 모두에서 CD-L1과 치수 MAE가 감소했으며, YCB-Video에서는 "
        "표의 네 지표가 모두 개선되었다."
        if korean
        else "CD-L1 and Dimension MAE improve on all three datasets. On YCB-Video, "
        "all four listed metrics improve."
    )
    sections.append("### 통계 분석" if korean else "### Statistical Analysis")
    sections.append(
        table(
            ["Metric", "Mean Difference", "95% CI"],
            [
                [LABELS[r["metric"]], r["mean_difference"], f"[{r['ci95_low']}, {r['ci95_high']}]"]
                for r in rows(root, "paper_bootstrap.csv")
            ],
        )
    )
    sections.append(
        "차이는 저장 VEIL-Net − SnowflakeNet FT이다. Paired scene-level bootstrap에서 "
        "F@0.03과 치수 MAE의 95% 신뢰구간은 0을 포함하지 않는다. 데이터셋별 F@0.03의 "
        "가중 평균과 bootstrap 평균 차이는 저장본 **0.65844**와 연결되며, 외부 보고값 "
        "**0.72843**에 대한 통계 검증으로 사용하지 않는다."
        if korean
        else "Differences are saved VEIL-Net minus SnowflakeNet FT. Paired scene-level "
        "bootstrap intervals exclude zero for F@0.03 and Dimension MAE. The "
        "dataset-weighted F@0.03 and bootstrap mean difference correspond to the "
        "saved score **0.65844**, not the external score **0.72843**."
    )
    return "\n\n".join(sections)


def update_document(path, content):
    text = path.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(f"Expected one results marker pair in {path}")
    before, tail = text.split(START)
    _, after = tail.split(END)
    path.write_text(before + START + "\n\n" + content + "\n\n" + END + after, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, help="Refresh an exported model card and include result tables"
    )
    args = parser.parse_args()
    english = render(ROOT)
    for name in ("docs/RESULTS.md", "benchmarks/README.md", "veil_net/model_card.md"):
        update_document(ROOT / name, english)
    update_document(ROOT / "README_KO.md", render(ROOT, korean=True))
    if args.model_dir:
        provenance = json.loads((args.model_dir / "provenance.json").read_text(encoding="utf-8"))
        card = (ROOT / "veil_net/model_card.md").read_text(encoding="utf-8")
        card += "\n## Export Provenance\n\n```json\n" + json.dumps(provenance, indent=2) + "\n```\n"
        (args.model_dir / "README.md").write_text(card, encoding="utf-8")
        destination = args.model_dir / "benchmarks"
        destination.mkdir(exist_ok=True)
        for source in (ROOT / "benchmarks").iterdir():
            if source.is_file() and source.suffix in {".csv", ".json", ".md"}:
                shutil.copy2(source, destination / source.name)
    print("Rendered all six manuscript tables in four public documents")


if __name__ == "__main__":
    main()
