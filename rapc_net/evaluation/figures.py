from __future__ import annotations

from pathlib import Path

import numpy as np

from rapc_net.utils.io import read_csv_dicts


def make_basic_figures(project_root: Path) -> int:
    import matplotlib.pyplot as plt

    figures = project_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    count = 0
    completion = project_root / "tables" / "completion_main.csv"
    if completion.exists():
        rows = read_csv_dicts(completion)
        if rows:
            labels = [r["model"] for r in rows]
            cd = [float(r["cd_l1_mean"]) for r in rows]
            f = [float(r["f_score_mean"]) for r in rows]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(labels, cd, color=["#6b7280" if label != "rapc_net" else "#d97706" for label in labels])
            ax.set_ylabel("CD-L1 lower is better")
            ax.set_title("Completion Accuracy")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figures / f"main_completion_cd_l1.{ext}", bbox_inches="tight", dpi=220)
                count += 1
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(labels, f, color=["#6b7280" if label != "rapc_net" else "#2563eb" for label in labels])
            ax.set_ylabel("F-score higher is better")
            ax.set_title("Completion F-score")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figures / f"main_completion_f_score.{ext}", bbox_inches="tight", dpi=220)
                count += 1
            plt.close(fig)

    robustness = project_root / "tables" / "robustness.csv"
    if robustness.exists():
        rows = [r for r in read_csv_dicts(robustness) if r.get("cd_l1_mean") not in {"", "nan"}]
        if rows:
            labels = [f"{r['degradation']}\n{r['severity']}" for r in rows]
            vals = [float(r["cd_l1_mean"]) for r in rows]
            fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.55), 4))
            ax.bar(labels, vals, color="#0f766e")
            ax.set_ylabel("CD-L1 lower is better")
            ax.set_title("Robustness Under Sensor Degradation")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.25)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figures / f"robustness_cd_l1.{ext}", bbox_inches="tight", dpi=220)
                count += 1
            plt.close(fig)

    robot = project_root / "tables" / "robot_feasibility_proxy.csv"
    if robot.exists():
        rows = [r for r in read_csv_dicts(robot) if r.get("grasp_feasibility_proxy_mean") not in {"", "nan"}]
        if rows:
            labels = [r["model"] for r in rows]
            feasibility = [float(r["grasp_feasibility_proxy_mean"]) for r in rows]
            collision = [float(r["hallucination_collision_proxy_mean"]) for r in rows]
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(labels))
            ax.bar(x - 0.18, feasibility, width=0.36, label="feasibility", color="#2563eb")
            ax.bar(x + 0.18, collision, width=0.36, label="collision proxy", color="#dc2626")
            ax.set_xticks(x, labels, rotation=25)
            ax.set_ylim(0, 1)
            ax.set_title("Robot-Relevance Proxy")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figures / f"robot_feasibility_proxy.{ext}", bbox_inches="tight", dpi=220)
                count += 1
            plt.close(fig)

    per_object = project_root / "tables" / "per_object.csv"
    if per_object.exists():
        rows = [r for r in read_csv_dicts(per_object) if r.get("risk_high_ratio") not in {"", "nan"}]
        if rows:
            rows = sorted(rows, key=lambda r: float(r["risk_high_ratio"]), reverse=True)[:10]
            labels = [f"{r['dataset']}-{r['object_id']}" for r in rows]
            vals = [float(r["risk_high_ratio"]) for r in rows]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(np.arange(len(labels)), vals, color="#f59e0b")
            ax.set_yticks(np.arange(len(labels)), labels)
            ax.invert_yaxis()
            ax.set_xlim(0, max(1.0, max(vals) * 1.05))
            ax.set_xlabel("High-risk point ratio")
            ax.set_title("Most Risk-Sensitive Objects")
            ax.grid(axis="x", alpha=0.25)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(figures / f"per_object_risk_high_ratio.{ext}", bbox_inches="tight", dpi=220)
                count += 1
            plt.close(fig)

    if count == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No executed metric tables found", ha="center", va="center")
        ax.set_axis_off()
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figures / f"missing_results_notice.{ext}", bbox_inches="tight", dpi=220)
            count += 1
        plt.close(fig)
    return count
