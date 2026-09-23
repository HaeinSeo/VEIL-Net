from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rapc_net.evaluation.metrics import nearest_distances
from rapc_net.evaluation.runners import load_compatible_model_state
from rapc_net.models import RAPCNet, RAPCNetConfig
from rapc_net.utils.io import read_csv_dicts, write_csv_dicts, write_json
from rapc_net.utils.pointcloud import sample_points, validate_points


COLORS = {
    "observed": (40, 105, 220),
    "prediction": (235, 132, 35),
    "ground_truth": (150, 150, 150),
    "missing": (35, 170, 90),
    "error": (220, 45, 45),
    "risk": (245, 205, 40),
    "uncertainty": (125, 70, 190),
}


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | tuple[int, int, int]) -> None:
    pts = validate_points(points)
    if isinstance(colors, tuple):
        cols = np.tile(np.asarray(colors, dtype=np.uint8), (len(pts), 1))
    else:
        cols = np.asarray(colors, dtype=np.uint8)
        if cols.shape != (len(pts), 3):
            raise ValueError(f"colors must be [N, 3], got {cols.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def scalar_to_color(values: np.ndarray, low: tuple[int, int, int], high: tuple[int, int, int]) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    mn, mx = float(np.nanmin(vals)), float(np.nanmax(vals))
    denom = max(mx - mn, 1e-8)
    t = np.clip((vals - mn) / denom, 0.0, 1.0)[:, None]
    lo = np.asarray(low, dtype=np.float32)
    hi = np.asarray(high, dtype=np.float32)
    return (lo * (1.0 - t) + hi * t).astype(np.uint8)


def select_samples_from_metrics(project_root: Path, model: str = "rapc_net", max_samples: int = 6) -> list[str]:
    metrics_path = project_root / "results" / "full" / "completion" / "completion_samples.csv"
    if not metrics_path.exists():
        pred_dir = project_root / "predictions" / model / "test"
        return [p.stem for p in sorted(pred_dir.rglob("*.npz"))[:max_samples]]
    rows = [r for r in read_csv_dicts(metrics_path) if r.get("model") == model]
    if not rows:
        return []
    rows.sort(key=lambda r: float(r["cd_l1"]))
    picks = [rows[0], rows[len(rows) // 2], rows[-1]]
    labels = ["best", "median", "worst"]
    selected = []
    for label, row in zip(labels, picks):
        selected.append({"selection": label, "sample_id": row["sample_id"], "cd_l1": row["cd_l1"], "f_score": row["f_score"]})
    extra = rows[1 : max(1, max_samples - len(selected) + 1)]
    for row in extra:
        selected.append({"selection": "additional", "sample_id": row["sample_id"], "cd_l1": row["cd_l1"], "f_score": row["f_score"]})
    write_csv_dicts(project_root / "visualizations" / "samples" / "selected_samples.csv", selected)
    return [r["sample_id"] for r in selected[:max_samples]]


def load_rapc_model(project_root: Path, seed: int, device: str) -> RAPCNet:
    ckpt = project_root / "checkpoints" / "rapc_net" / f"seed_{seed}" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"RAPC-Net checkpoint not found: {ckpt}")
    model = RAPCNet(RAPCNetConfig(output_points=16384)).to(device)
    state = torch.load(ckpt, map_location=device)
    load_report = load_compatible_model_state(model, state["model"])
    if load_report["skipped"] or load_report["missing"]:
        print(f"WARNING: loaded compatible RAPC-Net checkpoint weights only: {load_report}")
    model.eval()
    return model


def recompute_rapc_outputs(model: RAPCNet, partial: np.ndarray, device: str) -> dict[str, np.ndarray]:
    tensor = torch.from_numpy(partial.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return {k: v.squeeze(0).detach().cpu().numpy() for k, v in out.items() if torch.is_tensor(v)}


def render_static_views(out_base: Path, clouds: dict[str, np.ndarray], scalars: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8))
    panels = [
        ("Partial Input", "partial", COLORS["observed"]),
        ("Ground Truth", "gt", COLORS["ground_truth"]),
        ("RAPC-Net Completion", "prediction", COLORS["prediction"]),
        ("Prediction Error", "prediction", None),
        ("Point Risk", "partial", None),
        ("Uncertainty", "prediction", None),
    ]
    for idx, (title, key, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        pts = clouds[key]
        if len(pts) > 4096:
            pts = sample_points(pts, 4096, idx)
        if title == "Prediction Error":
            vals = scalars["pred_error"]
            if len(vals) != len(clouds[key]):
                vals = np.resize(vals, len(clouds[key]))
            vals = vals[: len(pts)]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap="Reds", s=1)
        elif title == "Point Risk":
            vals = scalars["risk"]
            vals = np.resize(vals, len(clouds[key]))[: len(pts)]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap="YlOrRd", s=2)
        elif title == "Uncertainty":
            vals = scalars["uncertainty"]
            vals = np.resize(vals, len(clouds[key]))[: len(pts)]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap="Purples", s=1)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=np.asarray(color) / 255.0, s=1)
        ax.set_title(title)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def render_model_comparison(out_base: Path, sample_id: str, panel_clouds: list[tuple[str, np.ndarray, tuple[int, int, int] | None]]) -> None:
    import math
    import matplotlib.pyplot as plt

    cols = min(4, max(1, len(panel_clouds)))
    rows = int(math.ceil(len(panel_clouds) / cols))
    fig = plt.figure(figsize=(4.0 * cols, 3.6 * rows))
    for idx, (title, points, color) in enumerate(panel_clouds, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        pts = points
        if len(pts) > 4096:
            pts = sample_points(pts, 4096, idx)
        if color is None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=np.linalg.norm(pts - pts.mean(axis=0), axis=1), cmap="viridis", s=1)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=np.asarray(color) / 255.0, s=1)
        ax.set_title(title)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-60)
    fig.suptitle(sample_id)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=220)
    plt.close(fig)


def make_interactive_html(path: Path, sample_id: str, clouds: dict[str, np.ndarray], scalars: dict[str, np.ndarray]) -> None:
    def arr(points: np.ndarray, max_points: int = 4096) -> list[list[float]]:
        pts = points
        if len(pts) > max_points:
            pts = sample_points(pts, max_points, 0)
        return np.round(pts, 6).tolist()

    payload = {
        "sample_id": sample_id,
        "partial": arr(clouds["partial"]),
        "prediction": arr(clouds["prediction"]),
        "ground_truth": arr(clouds["gt"]),
        "risk": np.round(np.resize(scalars["risk"], len(arr(clouds["partial"]))), 6).tolist(),
        "uncertainty": np.round(np.resize(scalars["uncertainty"], len(arr(clouds["prediction"]))), 6).tolist(),
    }
    script_data = json.dumps(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(sample_id)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body><div id="plot" style="width:100vw;height:100vh"></div>
<script>
const data = {script_data};
function trace(name, pts, color, size) {{
  return {{type:'scatter3d', mode:'markers', name, x:pts.map(p=>p[0]), y:pts.map(p=>p[1]), z:pts.map(p=>p[2]), marker:{{size, color}}}};
}}
const traces = [
  trace('partial observed', data.partial, 'rgb(40,105,220)', 2),
  trace('prediction complete', data.prediction, 'rgb(235,132,35)', 1.5),
  trace('ground truth', data.ground_truth, 'rgb(150,150,150)', 1.5),
  {{type:'scatter3d', mode:'markers', name:'risk segmentation', x:data.partial.map(p=>p[0]), y:data.partial.map(p=>p[1]), z:data.partial.map(p=>p[2]), marker:{{size:3, color:data.risk, colorscale:'YlOrRd', colorbar:{{title:'risk'}}}}, visible:'legendonly'}},
  {{type:'scatter3d', mode:'markers', name:'uncertainty', x:data.prediction.map(p=>p[0]), y:data.prediction.map(p=>p[1]), z:data.prediction.map(p=>p[2]), marker:{{size:2, color:data.uncertainty, colorscale:'Purples', colorbar:{{title:'unc'}}}}, visible:'legendonly'}}
];
Plotly.newPlot('plot', traces, {{title:data.sample_id, scene:{{aspectmode:'data'}}, margin:{{l:0,r:0,b:0,t:40}}}});
</script></body></html>""",
        encoding="utf-8",
    )


def export_visualizations(project_root: Path, seed: int, split: str, max_samples: int, device: str) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    sample_ids = select_samples_from_metrics(project_root, "rapc_net", max_samples)
    if not sample_ids:
        raise FileNotFoundError("No evaluated RAPC-Net samples found. Run 18_evaluate_completion_full.py first.")
    model = load_rapc_model(project_root, seed, device)
    pred_dir = project_root / "predictions" / "rapc_net" / split
    exported = []
    missing_predictions: list[dict[str, str]] = []
    for sample_id in sample_ids:
        pred_path = pred_dir / f"{sample_id}.npz"
        if not pred_path.exists():
            matches = list(pred_dir.rglob(f"{sample_id}.npz"))
            if not matches:
                continue
            pred_path = matches[0]
        data = np.load(pred_path)
        partial = data["input_partial"].astype(np.float32)
        gt = data["ground_truth"].astype(np.float32)
        rapc = recompute_rapc_outputs(model, partial, device)
        pred = rapc["completed_points"].astype(np.float32)
        risk = rapc["point_risk"].astype(np.float32)
        unc = rapc["uncertainty"].astype(np.float32)
        pred_error = nearest_distances(pred, gt)
        out_root = project_root / "visualizations" / "samples" / sample_id
        write_ply(out_root / "partial_observed.ply", partial, COLORS["observed"])
        write_ply(out_root / "ground_truth.ply", gt, COLORS["ground_truth"])
        write_ply(out_root / "rapc_prediction.ply", pred, COLORS["prediction"])
        write_ply(out_root / "risk_segmentation.ply", partial, scalar_to_color(risk, COLORS["observed"], COLORS["risk"]))
        write_ply(out_root / "uncertainty_map.ply", pred, scalar_to_color(unc, COLORS["prediction"], COLORS["uncertainty"]))
        write_ply(out_root / "error_heatmap.ply", pred, scalar_to_color(pred_error, COLORS["prediction"], COLORS["error"]))
        render_static_views(out_root / "paper_panel", {"partial": partial, "gt": gt, "prediction": pred}, {"risk": risk, "uncertainty": unc, "pred_error": pred_error})
        comparison_panels: list[tuple[str, np.ndarray, tuple[int, int, int] | None]] = [
            ("Partial Input", partial, COLORS["observed"]),
            ("CAD Ground Truth", gt, COLORS["ground_truth"]),
        ]
        for model_name, title in [
            ("fusion", "Geometric Fusion"),
            ("snowflakenet", "SnowflakeNet"),
            ("pointr", "PoinTr"),
            ("attriflow", "3DAttriFlow-PC"),
            ("rapc_net", "RAPC-Net"),
        ]:
            if model_name == "rapc_net":
                comparison_panels.append((title, pred, COLORS["prediction"]))
                continue
            candidates = list((project_root / "predictions" / model_name / split).rglob(f"{sample_id}.npz"))
            if not candidates:
                missing_predictions.append({"sample_id": sample_id, "model": model_name, "reason": "prediction file not found"})
                continue
            model_data = np.load(candidates[0])
            if "prediction_complete" not in model_data.files:
                missing_predictions.append({"sample_id": sample_id, "model": model_name, "reason": "prediction_complete field missing"})
                continue
            comparison_panels.append((title, model_data["prediction_complete"].astype(np.float32), COLORS["prediction"]))
        render_model_comparison(out_root / "model_comparison", sample_id, comparison_panels)
        make_interactive_html(out_root / "interactive.html", sample_id, {"partial": partial, "gt": gt, "prediction": pred}, {"risk": risk, "uncertainty": unc})
        write_json(
            out_root / "metadata.json",
            {
                "sample_id": sample_id,
                "split": split,
                "seed": seed,
                "device": device,
                "files": ["partial_observed.ply", "ground_truth.ply", "rapc_prediction.ply", "risk_segmentation.ply", "uncertainty_map.ply", "error_heatmap.ply", "paper_panel.png", "paper_panel.pdf", "paper_panel.svg", "model_comparison.png", "model_comparison.pdf", "model_comparison.svg", "interactive.html"],
            },
        )
        exported.append({"sample_id": sample_id, "visualization_dir": str(out_root), "mean_pred_error": float(pred_error.mean()), "mean_risk": float(risk.mean()), "mean_uncertainty": float(unc.mean())})
    write_csv_dicts(project_root / "visualizations" / "samples" / "visualization_index.csv", exported)
    write_csv_dicts(project_root / "visualizations" / "samples" / "missing_predictions.csv", missing_predictions, fieldnames=["sample_id", "model", "reason"])
    return {"samples": len(exported), "index": str(project_root / "visualizations" / "samples" / "visualization_index.csv")}
