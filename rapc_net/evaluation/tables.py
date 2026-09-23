from __future__ import annotations

from pathlib import Path

from rapc_net.utils.io import read_csv_dicts, write_csv_dicts


def build_icra_tables(project_root: Path) -> int:
    tables = project_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    completion = tables / "completion_main.csv"
    rows = read_csv_dicts(completion) if completion.exists() else []
    if rows:
        write_csv_dicts(tables / "main_completion.csv", rows)
    ablation_rows = []
    for path in sorted(tables.glob("ablation_*.csv")):
        ablation_rows.extend(read_csv_dicts(path))
    if ablation_rows:
        write_csv_dicts(tables / "ablation.csv", ablation_rows)
    table_names = ["main_completion", "robustness", "cross_dataset", "per_object", "pose_refinement", "robot_feasibility_proxy", "ablation", "efficiency"]
    for name in table_names:
        path = tables / f"{name}.csv"
        if path.exists():
            continue
        write_csv_dicts(path, [{"model": "", "status": "not_run"}], fieldnames=["model", "status"])
    return len(table_names)
