from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rapc_net.evaluation.metrics import evaluate_completion, evaluate_completion_torch, nearest_distances
from rapc_net.models import RAPCNet, RAPCNetConfig
from rapc_net.pose.refinement import rigid_align_svd
from rapc_net.training.trainer import create_rapc_variant
from rapc_net.utils.io import read_csv_dicts, read_json, write_csv_dicts, write_json, write_text
from rapc_net.utils.pointcloud import sample_points


def load_compatible_model_state(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    current = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in current and current[k].shape == v.shape}
    skipped = len(state_dict) - len(compatible)
    missing = len(current) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "skipped": skipped, "missing": missing}


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _prediction_payload(partial: np.ndarray, gt: np.ndarray, outputs: dict[str, torch.Tensor], elapsed_ms: float) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "input_partial": partial.astype(np.float32),
        "prediction_complete": outputs["completed_points"].squeeze(0).detach().cpu().numpy().astype(np.float32),
        "ground_truth": gt.astype(np.float32),
        "inference_time_ms": np.asarray([elapsed_ms], dtype=np.float32),
    }
    for key in ("generated_points", "point_risk", "predicted_risk", "uncertainty", "point_uncertainty", "gate_score", "global_confidence", "route_weights"):
        if key in outputs:
            payload[key] = outputs[key].squeeze(0).detach().cpu().numpy().astype(np.float32)
    for key in ('input_source_indices', 'input_inlier_mask', 'processed_partial'):
        if key in outputs:
            payload[key] = outputs[key].squeeze(0).detach().cpu().numpy()
    if 'input_source_indices' in payload:
        source = payload['input_source_indices']
        counts = np.bincount(source, minlength=len(partial))
        for key in ('point_risk', 'predicted_risk'):
            if key in payload:
                payload['processed_' + key] = payload[key]
                aligned = np.full(len(partial), np.nan, dtype=np.float32)
                sums = np.bincount(source, weights=payload[key], minlength=len(partial))
                aligned[counts > 0] = sums[counts > 0] / counts[counts > 0]
                payload[key] = aligned
    return payload


def _common_prediction_paths(project_root: Path, split: str, models: list[str], max_samples: int | None = None) -> dict[str, list[Path]]:
    model_paths: dict[str, dict[str, Path]] = {}
    for model in models:
        pred_dir = project_root / "predictions" / model / split
        if not pred_dir.exists():
            continue
        paths_by_sample: dict[str, Path] = {}
        for path in sorted(pred_dir.rglob("*.npz")):
            paths_by_sample.setdefault(path.stem, path)
        if paths_by_sample:
            model_paths[model] = paths_by_sample
    if not model_paths:
        return {}
    common_ids = sorted(set.intersection(*(set(paths.keys()) for paths in model_paths.values())))
    if max_samples is not None:
        common_ids = common_ids[:max_samples]
    return {model: [paths[sample_id] for sample_id in common_ids] for model, paths in model_paths.items()}


def export_rapc_like_predictions_from_checkpoint(
    project_root: Path,
    split: str,
    seed: int,
    device: str,
    model_key: str,
    checkpoint_key: str,
    variant: str | None = None,
    max_samples: int | None = None,
) -> int:
    pair_root = project_root / "cache" / "completion_pairs" / split
    if not pair_root.exists():
        raise FileNotFoundError(f"{pair_root} is missing; run 07_prepare_completion_pairs.py --split {split} first")
    ckpt = project_root / "checkpoints" / checkpoint_key / f"seed_{seed}" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} is missing; train {checkpoint_key} first")
    paths = sorted(pair_root.rglob("*.npz"))
    if max_samples is not None:
        paths = paths[:max_samples]
    model = create_rapc_variant(variant, output_points=16384).to(device)
    state = torch.load(ckpt, map_location=device)
    model.config = replace(model.config, bbox_margin=float(state.get("model_config", {}).get("bbox_margin", model.config.bbox_margin)))
    model.config = replace(model.config, uncertainty_moves_points=state.get("model_config", {}).get("uncertainty_moves_points", True))
    if state.get("model_config", {}).get("use_surface_refiner", False):
        model = RAPCNet(replace(model.config, use_surface_refiner=True)).to(device)
    if state.get("model_config", {}).get("use_spatial_transformer", False):
        from rapc_net.models.rapc_net import RAPCNetConfig
        model = RAPCNet(RAPCNetConfig(**state["model_config"])).to(device)
        model.load_state_dict(state["model"], strict=True)
    load_report = load_compatible_model_state(model, state["model"])
    if load_report["skipped"] or load_report["missing"]:
        print(f"WARNING: loaded compatible RAPC-Net checkpoint weights only: {load_report}")
    model.eval()
    out_dir = project_root / "predictions" / model_key / split
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with torch.no_grad():
        for path in paths:
            data = np.load(path)
            if "complete" not in data.files:
                continue
            partial = data["partial"].astype(np.float32)
            tensor = torch.from_numpy(partial).unsqueeze(0).to(device)
            risk_tensor = torch.from_numpy(data["risk_target"].astype(np.float32)).unsqueeze(0).to(device) if variant == "A7" and "risk_target" in data.files else None
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            import time

            start = time.perf_counter()
            outputs = model(tensor, risk_tensor)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            payload = _prediction_payload(partial, data["complete"].astype(np.float32), outputs, (time.perf_counter() - start) * 1000.0)
            np.savez_compressed(out_dir / path.name, **payload)
            count += 1
    return count


def export_rapc_predictions_from_checkpoint(
    project_root: Path,
    split: str,
    seed: int,
    device: str,
    max_samples: int | None = None,
) -> int:
    return export_rapc_like_predictions_from_checkpoint(project_root, split, seed, device, "rapc_net", "rapc_net", None, max_samples)


def export_observed_upsample_predictions(
    project_root: Path,
    split: str,
    max_samples: int | None = None,
    output_points: int = 16384,
) -> int:
    pair_root = project_root / "cache" / "completion_pairs" / split
    if not pair_root.exists():
        raise FileNotFoundError(f"{pair_root} is missing; run 07_prepare_completion_pairs.py --split {split} first")
    paths = sorted(pair_root.rglob("*.npz"))
    if max_samples is not None:
        paths = paths[:max_samples]
    out_dir = project_root / "predictions" / "fusion" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, path in enumerate(paths):
        data = np.load(path)
        if "complete" not in data.files:
            continue
        partial = data["partial"].astype(np.float32)
        pred = sample_points(partial, output_points, seed=idx)
        np.savez_compressed(
            out_dir / path.name,
            input_partial=partial,
            prediction_complete=pred.astype(np.float32),
            ground_truth=data["complete"].astype(np.float32),
            inference_time_ms=np.asarray([0.0], dtype=np.float32),
            baseline_note=np.asarray(["observed_surface_upsample"], dtype=str),
        )
        count += 1
    return count


def evaluate_prediction_dir(
    project_root: Path,
    split: str,
    models: list[str],
    device: str = "cuda",
    max_samples: int | None = None,
    eval_max_points: int | None = 4096,
    eval_chunk: int = 2048,
    samples_path: Path | None = None,
    summary_path: Path | None = None,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    common_paths = _common_prediction_paths(project_root, split, models, max_samples=max_samples)
    if sample_ids is not None:
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError('Expected nonempty unique evaluation sample IDs')
        common_paths = {}
        for model in models:
            indexed = {}
            for path in (project_root / 'predictions' / model / split).rglob('*.npz'):
                if path.stem in indexed:
                    raise ValueError(f'Duplicate prediction: {model}/{path.stem}')
                indexed[path.stem] = path
            missing = set(sample_ids) - indexed.keys()
            if missing:
                raise ValueError(f'{model}/{split}: missing {len(missing)} required predictions')
            common_paths[model] = [indexed[sid] for sid in sorted(sample_ids)]
    if not common_paths:
        write_csv_dicts(samples_path or project_root / "results" / "full" / "completion" / "completion_samples.csv", rows)
        write_csv_dicts(summary_path or project_root / "tables" / "completion_main.csv", [])
        return {"samples": 0, "models": []}
    for model in models:
        if model not in common_paths:
            continue
        paths = common_paths[model]
        for path_index, path in enumerate(paths):
            data = np.load(path)
            required = {"prediction_complete", "ground_truth", "input_partial"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path} missing fields: {sorted(missing)}")
            if sample_ids is not None and model != models[0]:
                reference_path = common_paths[models[0]][path_index]
                with np.load(reference_path) as reference:
                    for key in ('ground_truth', 'input_partial'):
                        if not np.array_equal(data[key], reference[key]):
                            raise ValueError(f'{model}/{path.stem}: {key} differs across models')
            if device.startswith("cuda") and torch.cuda.is_available():
                metrics = evaluate_completion_torch(
                    data["prediction_complete"],
                    data["ground_truth"],
                    data["input_partial"],
                    device=device,
                    max_points=eval_max_points,
                    chunk=eval_chunk,
                ).as_dict()
                evaluator = "torch_gpu"
            else:
                metrics = evaluate_completion(data["prediction_complete"], data["ground_truth"], data["input_partial"]).as_dict()
                evaluator = "numpy_cpu"
            if "uncertainty" in data.files:
                pred_err = nearest_distances(data["prediction_complete"], data["ground_truth"])
                metrics["uncertainty_error_corr"] = _safe_corr(data["uncertainty"], pred_err)
                metrics["uncertainty_mean"] = float(np.asarray(data["uncertainty"], dtype=np.float32).mean())
            if "point_risk" in data.files:
                metrics["risk_mean"] = float(np.asarray(data["point_risk"], dtype=np.float32).mean())
            rows.append({"model": model, "sample_id": path.stem, "evaluator": evaluator, "eval_max_points": eval_max_points or "all", **metrics})
    out_dir = project_root / "results" / "full" / "completion"
    write_csv_dicts(samples_path or out_dir / "completion_samples.csv", rows)
    summary: list[dict[str, Any]] = []
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        if not subset:
            continue
        summary_row = {
            "model": model,
            "samples": len(subset),
            "cd_l1_mean": float(np.mean([float(r["cd_l1"]) for r in subset])),
            "cd_l2_mean": float(np.mean([float(r["cd_l2"]) for r in subset])),
            "f_score_mean": float(np.mean([float(r["f_score"]) for r in subset])),
            "f_score_001_mean": float(np.mean([float(r["f_score_001"]) for r in subset])),
            "f_score_003_mean": float(np.mean([float(r["f_score_003"]) for r in subset])),
            "f_score_005_mean": float(np.mean([float(r["f_score_005"]) for r in subset])),
            "f_score_010_mean": float(np.mean([float(r["f_score_010"]) for r in subset])),
            "f_score_020_mean": float(np.mean([float(r["f_score_020"]) for r in subset])),
            "f_score_050_mean": float(np.mean([float(r["f_score_050"]) for r in subset])),
            "missing_region_cd_mean": float(np.nanmean([float(r["missing_region_cd"]) for r in subset])),
            "observed_region_error_mean": float(np.nanmean([float(r["observed_region_error"]) for r in subset])),
            "dimension_mae_mean": float(np.nanmean([float(r["dimension_mae"]) for r in subset])),
        }
        for key in ("uncertainty_error_corr", "uncertainty_mean", "risk_mean"):
            vals = [float(r[key]) for r in subset if key in r and str(r[key]) not in {"", "nan"}]
            if vals:
                summary_row[f"{key}_mean"] = float(np.mean(vals))
        summary.append(summary_row)
    write_csv_dicts(summary_path or project_root / "tables" / "completion_main.csv", summary)
    return {"samples": len(rows), "models": [s["model"] for s in summary]}


def evaluate_rapc_robustness(
    project_root: Path,
    split: str,
    seed: int,
    device: str,
    max_samples: int | None = None,
    eval_max_points: int | None = 4096,
    eval_chunk: int = 2048,
) -> dict[str, Any]:
    degraded_root = project_root / "cache" / "degradations" / split
    if not degraded_root.exists():
        raise FileNotFoundError(f"{degraded_root} is missing; run 19_generate_degradations.py --split {split} first")
    ckpt = project_root / "checkpoints" / "rapc_net" / f"seed_{seed}" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} is missing; train RAPC-Net first")
    paths = sorted(degraded_root.rglob("*.npz"))
    if max_samples is not None:
        paths = paths[:max_samples]
    model = create_rapc_variant(None, output_points=16384).to(device)
    state = torch.load(ckpt, map_location=device)
    load_report = load_compatible_model_state(model, state["model"])
    if load_report["skipped"] or load_report["missing"]:
        print(f"WARNING: loaded compatible RAPC-Net checkpoint weights only: {load_report}")
    model.eval()
    rows: list[dict[str, Any]] = []
    pred_root = project_root / "predictions" / "rapc_net_robustness" / split
    pred_root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for path in paths:
            degraded = np.load(path)
            source_pair = Path(str(degraded["source_pair"])) if "source_pair" in degraded.files else None
            if source_pair is None or not source_pair.exists():
                continue
            pair = np.load(source_pair)
            partial = degraded["partial"].astype(np.float32)
            tensor = torch.from_numpy(partial).unsqueeze(0).to(device)
            outputs = model(tensor)
            pred = outputs["completed_points"].squeeze(0).detach().cpu().numpy().astype(np.float32)
            metrics = evaluate_completion_torch(
                pred,
                pair["complete"],
                partial,
                device=device,
                max_points=eval_max_points,
                chunk=eval_chunk,
                seed=seed,
            ).as_dict()
            rel = path.relative_to(degraded_root)
            kind = rel.parts[0] if len(rel.parts) > 2 else "unknown"
            severity = rel.parts[1] if len(rel.parts) > 2 else "unknown"
            sample_id = path.stem
            out_dir = pred_root / kind / severity
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / path.name,
                input_partial=partial,
                prediction_complete=pred,
                ground_truth=pair["complete"].astype(np.float32),
                source_pair=str(source_pair),
            )
            rows.append({"model": "rapc_net", "sample_id": sample_id, "degradation": kind, "severity": severity, **metrics})
    out_dir = project_root / "results" / "full" / "robustness"
    write_csv_dicts(out_dir / "robustness_samples.csv", rows)
    clean_by_sample = {r["sample_id"]: float(r["cd_l1"]) for r in rows if r["degradation"] == "clean"}
    summary: list[dict[str, Any]] = []
    for kind in sorted({r["degradation"] for r in rows}):
        for severity in sorted({r["severity"] for r in rows if r["degradation"] == kind}):
            subset = [r for r in rows if r["degradation"] == kind and r["severity"] == severity]
            vals = [float(r["cd_l1"]) for r in subset]
            drops = [float(r["cd_l1"]) - clean_by_sample[r["sample_id"]] for r in subset if r["sample_id"] in clean_by_sample]
            summary.append(
                {
                    "model": "rapc_net",
                    "degradation": kind,
                    "severity": severity,
                    "samples": len(subset),
                    "cd_l1_mean": float(np.mean(vals)) if vals else float("nan"),
                    "cd_l1_drop_vs_clean": float(np.mean(drops)) if drops else float("nan"),
                    "f_score_mean": float(np.mean([float(r["f_score"]) for r in subset])) if subset else float("nan"),
                }
            )
    write_csv_dicts(project_root / "tables" / "robustness.csv", summary)
    return {"samples": len(rows), "groups": len(summary)}


def evaluate_cross_dataset_summary(project_root: Path, split: str = "test") -> dict[str, Any]:
    path = project_root / "results" / "full" / "completion" / "completion_samples.csv"
    if not path.exists():
        raise FileNotFoundError("completion sample results are missing; run 18_evaluate_completion_full.py first")
    rows = read_csv_dicts(path)
    summary: list[dict[str, Any]] = []
    for model in sorted({r["model"] for r in rows}):
        model_rows = [r for r in rows if r["model"] == model]
        datasets = sorted({r["sample_id"].split("_")[0] for r in model_rows})
        for dataset in datasets:
            subset = [r for r in model_rows if r["sample_id"].startswith(dataset + "_")]
            vals = [float(r["cd_l1"]) for r in subset]
            summary.append(
                {
                    "model": model,
                    "split": split,
                    "dataset": dataset,
                    "samples": len(subset),
                    "cd_l1_mean": float(np.mean(vals)) if vals else float("nan"),
                    "f_score_mean": float(np.mean([float(r["f_score"]) for r in subset])) if subset else float("nan"),
                    "missing_region_cd_mean": float(np.mean([float(r["missing_region_cd"]) for r in subset if r["missing_region_cd"] not in {"", "nan"}])) if subset else float("nan"),
                }
            )
    write_csv_dicts(project_root / "tables" / "cross_dataset.csv", summary)
    write_csv_dicts(project_root / "results" / "full" / "cross_dataset" / "cross_dataset_summary.csv", summary)
    return {"rows": len(summary), "models": sorted({r["model"] for r in rows})}


def evaluate_pose_refinement_proxy(
    project_root: Path,
    split: str,
    models: list[str],
    max_samples: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    common_paths = _common_prediction_paths(project_root, split, models, max_samples=max_samples)
    for model in models:
        if model not in common_paths:
            continue
        paths = common_paths[model]
        for path in paths:
            data = np.load(path)
            if not {"prediction_complete", "ground_truth"}.issubset(data.files):
                continue
            pred = data["prediction_complete"].astype(np.float32)
            gt = data["ground_truth"].astype(np.float32)
            count = min(2048, len(pred), len(gt))
            pred_s = sample_points(pred, count, seed=17)
            gt_s = sample_points(gt, count, seed=18)
            nearest = []
            for start in range(0, len(pred_s), 512):
                block = pred_s[start : start + 512]
                dist = ((block[:, None, :] - gt_s[None, :, :]) ** 2).sum(axis=2)
                nearest.append(gt_s[dist.argmin(axis=1)])
            paired_gt = np.concatenate(nearest, axis=0).astype(np.float32)
            transform = rigid_align_svd(pred_s, paired_gt)
            aligned_s = (np.concatenate([pred_s, np.ones((len(pred_s), 1), dtype=np.float32)], axis=1) @ transform.T)[:, :3].astype(np.float32)
            before = evaluate_completion(pred_s, gt_s, None).as_dict()
            after = evaluate_completion(aligned_s, gt_s, None).as_dict()
            rows.append(
                {
                    "model": model,
                    "sample_id": path.stem,
                    "cd_l1_before": before["cd_l1"],
                    "cd_l1_after_rigid_refine": after["cd_l1"],
                    "cd_l1_delta": before["cd_l1"] - after["cd_l1"],
                    "dimension_mae_before": before["dimension_mae"],
                    "dimension_mae_after": after["dimension_mae"],
                }
            )
    write_csv_dicts(project_root / "results" / "full" / "pose_refinement" / "pose_refinement_samples.csv", rows)
    summary = []
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model]
        summary.append(
            {
                "model": model,
                "samples": len(subset),
                "cd_l1_before": float(np.mean([float(r["cd_l1_before"]) for r in subset])),
                "cd_l1_after_rigid_refine": float(np.mean([float(r["cd_l1_after_rigid_refine"]) for r in subset])),
                "cd_l1_delta": float(np.mean([float(r["cd_l1_delta"]) for r in subset])),
            }
        )
    write_csv_dicts(project_root / "tables" / "pose_refinement.csv", summary)
    return {"samples": len(rows), "models": [r["model"] for r in summary]}


def evaluate_robot_feasibility_proxy(
    project_root: Path,
    split: str,
    models: list[str],
    max_samples: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    common_paths = _common_prediction_paths(project_root, split, models, max_samples=max_samples)
    for model in models:
        if model not in common_paths:
            continue
        paths = common_paths[model]
        for path in paths:
            data = np.load(path)
            if not {"prediction_complete", "ground_truth", "input_partial"}.issubset(data.files):
                continue
            pred = data["prediction_complete"].astype(np.float32)
            gt = data["ground_truth"].astype(np.float32)
            partial = data["input_partial"].astype(np.float32)
            pred_s = sample_points(pred, min(2048, len(pred)), seed=31)
            gt_s = sample_points(gt, min(2048, len(gt)), seed=32)
            partial_s = sample_points(partial, min(2048, len(partial)), seed=33)
            pred_to_gt = nearest_distances(pred_s, gt_s)
            partial_to_pred = nearest_distances(partial_s, pred_s)
            gt_to_partial = nearest_distances(gt_s, partial_s)
            missing_region = gt_s[gt_to_partial > 0.02]
            missing_coverage = 1.0
            if len(missing_region) > 0:
                missing_coverage = float((nearest_distances(missing_region, pred_s) < 0.03).mean())
            hallucination_rate = float((pred_to_gt > 0.05).mean())
            observed_preservation = float((partial_to_pred < 0.02).mean())
            span_pred = pred_s.max(axis=0) - pred_s.min(axis=0)
            span_gt = gt_s.max(axis=0) - gt_s.min(axis=0)
            dimension_stability = float(1.0 / (1.0 + np.abs(span_pred - span_gt).mean()))
            uncertainty_mean = float(np.asarray(data["uncertainty"], dtype=np.float32).mean()) if "uncertainty" in data.files else float("nan")
            risk_mean = float(np.asarray(data["point_risk"], dtype=np.float32).mean()) if "point_risk" in data.files else float("nan")
            feasibility = (
                0.35 * missing_coverage
                + 0.25 * observed_preservation
                + 0.20 * (1.0 - hallucination_rate)
                + 0.20 * dimension_stability
            )
            if not np.isnan(uncertainty_mean):
                feasibility *= max(0.0, 1.0 - 0.25 * uncertainty_mean)
            rows.append(
                {
                    "model": model,
                    "sample_id": path.stem,
                    "missing_region_coverage": missing_coverage,
                    "observed_preservation_rate": observed_preservation,
                    "hallucination_collision_proxy": hallucination_rate,
                    "dimension_stability": dimension_stability,
                    "uncertainty_mean": uncertainty_mean,
                    "risk_mean": risk_mean,
                    "grasp_feasibility_proxy": float(feasibility),
                }
            )
    write_csv_dicts(project_root / "results" / "full" / "robot_proxy" / "robot_proxy_samples.csv", rows)
    summary: list[dict[str, Any]] = []
    for model in sorted({r["model"] for r in rows}):
        subset = [r for r in rows if r["model"] == model]
        summary.append(
            {
                "model": model,
                "samples": len(subset),
                "grasp_feasibility_proxy_mean": float(np.mean([float(r["grasp_feasibility_proxy"]) for r in subset])),
                "hallucination_collision_proxy_mean": float(np.mean([float(r["hallucination_collision_proxy"]) for r in subset])),
                "observed_preservation_rate_mean": float(np.mean([float(r["observed_preservation_rate"]) for r in subset])),
                "missing_region_coverage_mean": float(np.mean([float(r["missing_region_coverage"]) for r in subset])),
            }
        )
    write_csv_dicts(project_root / "tables" / "robot_feasibility_proxy.csv", summary)
    return {"samples": len(rows), "models": [r["model"] for r in summary]}


def _manifest_lookup(project_root: Path, split: str) -> dict[str, dict[str, str]]:
    path = project_root / "manifests" / f"{split}.csv"
    if not path.exists():
        return {}
    return {row["sample_id"]: row for row in read_csv_dicts(path)}


def _pair_meta(project_root: Path, split: str, sample_id: str) -> dict[str, Any]:
    matches = list((project_root / "cache" / "completion_pairs" / split).rglob(f"{sample_id}.json"))
    if not matches:
        return {}
    try:
        meta = read_json(matches[0])
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}


def _object_label(meta: dict[str, Any]) -> str:
    dataset = str(meta.get("dataset", "unknown"))
    object_id = str(meta.get("object_id", "unknown"))
    cad_path = str(meta.get("cad_path", ""))
    cad_stem = Path(cad_path).stem if cad_path else ""
    if cad_stem:
        return f"{dataset}:{object_id} {cad_stem}"
    return f"{dataset}:{object_id}"


def evaluate_per_object_risk_analysis(
    project_root: Path,
    split: str,
    models: list[str],
    max_samples: int | None = None,
) -> dict[str, Any]:
    manifest = _manifest_lookup(project_root, split)
    sample_rows: list[dict[str, Any]] = []
    common_paths = _common_prediction_paths(project_root, split, models, max_samples=max_samples)
    for model in models:
        if model not in common_paths:
            continue
        paths = common_paths[model]
        for path in paths:
            data = np.load(path)
            if not {"prediction_complete", "ground_truth", "input_partial"}.issubset(data.files):
                continue
            sample_id = path.stem
            meta = {**_pair_meta(project_root, split, sample_id), **manifest.get(sample_id, {})}
            pred = data["prediction_complete"].astype(np.float32)
            gt = data["ground_truth"].astype(np.float32)
            partial = data["input_partial"].astype(np.float32)
            pred_s = sample_points(pred, min(2048, len(pred)), seed=41)
            gt_s = sample_points(gt, min(2048, len(gt)), seed=42)
            partial_s = sample_points(partial, min(2048, len(partial)), seed=43)
            completion_metrics = evaluate_completion(pred_s, gt_s, partial_s).as_dict()
            pred_err = nearest_distances(pred_s, gt_s)
            gt_to_partial = nearest_distances(gt_s, partial_s)
            gt_missing_ratio_002 = float((gt_to_partial > 0.02).mean())
            gt_missing_ratio_003 = float((gt_to_partial > 0.03).mean())
            gt_missing_ratio_005 = float((gt_to_partial > 0.05).mean())
            gt_missing_ratio_020 = float((gt_to_partial > 0.20).mean())
            gt_missing_ratio_050 = float((gt_to_partial > 0.50).mean())
            gt_missing_ratio_075 = float((gt_to_partial > 0.75).mean())
            missing = gt_s[gt_to_partial > 0.02]
            missing_cd = float(nearest_distances(missing, pred_s).mean()) if len(missing) else float("nan")
            risk = np.asarray(data["point_risk"], dtype=np.float32) if "point_risk" in data.files else np.empty((0,), dtype=np.float32)
            unc = np.asarray(data["uncertainty"], dtype=np.float32) if "uncertainty" in data.files else np.empty((0,), dtype=np.float32)
            sample_rows.append(
                {
                    "model": model,
                    "dataset": meta.get("dataset", sample_id.split("_")[0]),
                    "object_id": meta.get("object_id", "unknown"),
                    "object_label": _object_label(meta),
                    "scene_id": meta.get("scene_id", "unknown"),
                    "sample_id": sample_id,
                    "cd_l1_sampled": float(completion_metrics["cd_l1"]),
                    "cd_l2_sampled": float(completion_metrics["cd_l2"]),
                    "f_score_020_sampled": float(completion_metrics.get("f_score_020", float("nan"))),
                    "f_score_050_sampled": float(completion_metrics.get("f_score_050", float("nan"))),
                    "missing_region_cd_sampled": missing_cd,
                    "gt_to_input_dist_mean": float(gt_to_partial.mean()),
                    "gt_to_input_dist_p95": float(np.nanquantile(gt_to_partial, 0.95)),
                    "gt_missing_ratio_002": gt_missing_ratio_002,
                    "gt_missing_ratio_003": gt_missing_ratio_003,
                    "gt_missing_ratio_005": gt_missing_ratio_005,
                    "gt_missing_ratio_020": gt_missing_ratio_020,
                    "gt_missing_ratio_050": gt_missing_ratio_050,
                    "gt_missing_ratio_075": gt_missing_ratio_075,
                    "risk_mean": float(risk.mean()) if risk.size else float("nan"),
                    "risk_high_ratio": float((risk > 0.7).mean()) if risk.size else float("nan"),
                    "risk_dynamic_range": float(risk.max() - risk.min()) if risk.size else float("nan"),
                    "uncertainty_mean": float(unc.mean()) if unc.size else float("nan"),
                    "uncertainty_high_ratio": float((unc > 0.7).mean()) if unc.size else float("nan"),
                    "uncertainty_error_corr": _safe_corr(unc, pred_err) if unc.size else float("nan"),
                    "observed_preservation_error": float(nearest_distances(partial_s, pred_s).mean()),
                    "hallucination_rate": float((pred_err > 0.05).mean()),
                }
            )
    write_csv_dicts(project_root / "results" / "full" / "per_object" / "per_object_samples.csv", sample_rows)
    summary: list[dict[str, Any]] = []
    group_keys = sorted({(r["model"], r["dataset"], r["object_id"]) for r in sample_rows})

    def finite_mean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    for model, dataset, object_id in group_keys:
        subset = [r for r in sample_rows if r["model"] == model and r["dataset"] == dataset and r["object_id"] == object_id]
        summary.append(
            {
                "model": model,
                "dataset": dataset,
                "object_id": object_id,
                "object_label": subset[0].get("object_label", f"{dataset}:{object_id}"),
                "samples": len(subset),
                "cd_l1_mean": float(np.mean([float(r["cd_l1_sampled"]) for r in subset])),
                "cd_l2_mean": float(np.mean([float(r["cd_l2_sampled"]) for r in subset])),
                "f_score_020_mean": finite_mean([float(r["f_score_020_sampled"]) for r in subset]),
                "f_score_050_mean": finite_mean([float(r["f_score_050_sampled"]) for r in subset]),
                "missing_region_cd_mean": finite_mean([float(r["missing_region_cd_sampled"]) for r in subset]),
                "gt_to_input_dist_mean": finite_mean([float(r["gt_to_input_dist_mean"]) for r in subset]),
                "gt_to_input_dist_p95": finite_mean([float(r["gt_to_input_dist_p95"]) for r in subset]),
                "gt_missing_ratio_002": finite_mean([float(r["gt_missing_ratio_002"]) for r in subset]),
                "gt_missing_ratio_003": finite_mean([float(r["gt_missing_ratio_003"]) for r in subset]),
                "gt_missing_ratio_005": finite_mean([float(r["gt_missing_ratio_005"]) for r in subset]),
                "gt_missing_ratio_020": finite_mean([float(r["gt_missing_ratio_020"]) for r in subset]),
                "gt_missing_ratio_050": finite_mean([float(r["gt_missing_ratio_050"]) for r in subset]),
                "gt_missing_ratio_075": finite_mean([float(r["gt_missing_ratio_075"]) for r in subset]),
                "risk_mean": finite_mean([float(r["risk_mean"]) for r in subset]),
                "risk_high_ratio": finite_mean([float(r["risk_high_ratio"]) for r in subset]),
                "uncertainty_mean": finite_mean([float(r["uncertainty_mean"]) for r in subset]),
                "uncertainty_error_corr": finite_mean([float(r["uncertainty_error_corr"]) for r in subset]),
                "observed_preservation_error": float(np.mean([float(r["observed_preservation_error"]) for r in subset])),
                "hallucination_rate": float(np.mean([float(r["hallucination_rate"]) for r in subset])),
            }
        )
    summary.sort(key=lambda row: (str(row["model"]), -float(row["risk_high_ratio"]) if not math.isnan(float(row["risk_high_ratio"])) else -1.0, -float(row["cd_l1_mean"])))
    write_csv_dicts(project_root / "tables" / "per_object.csv", summary)
    return {"samples": len(sample_rows), "objects": len(summary), "models": sorted({r["model"] for r in sample_rows})}


def bootstrap_ci(values: list[float], seed: int = 0, rounds: int = 1000) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(rounds)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze_statistics(project_root: Path) -> dict[str, Any]:
    path = project_root / "results" / "full" / "completion" / "completion_samples.csv"
    if not path.exists():
        raise FileNotFoundError("completion sample results are missing; run 18_evaluate_completion_full.py first")
    rows = read_csv_dicts(path)
    out = []
    for model in sorted({r["model"] for r in rows}):
        vals = [float(r["cd_l1"]) for r in rows if r["model"] == model and r["cd_l1"] not in {"", "nan"}]
        lo, hi = bootstrap_ci(vals)
        out.append({"model": model, "n": len(vals), "cd_l1_mean": np.mean(vals) if vals else math.nan, "cd_l1_std": np.std(vals) if vals else math.nan, "ci95_low": lo, "ci95_high": hi})
    write_csv_dicts(project_root / "tables" / "statistics_summary.csv", out)
    return {"models": len(out), "rows": len(rows)}


def write_reproducibility_reports(project_root: Path) -> dict[str, Any]:
    required = ["configs", "manifests", "logs", "status"]
    checks = {name: (project_root / name).exists() for name in required}
    status_files = sorted((project_root / "status").glob("*.done")) if (project_root / "status").exists() else []
    lines = ["# Reproducibility Report", ""]
    lines += [f"- {k}: {v}" for k, v in checks.items()]
    lines += ["", "## Completed Steps", ""]
    lines += [f"- {p.stem}" for p in status_files] or ["- none"]
    write_text(project_root / "docs" / "REPRODUCIBILITY_REPORT.md", "\n".join(lines) + "\n")

    report = ["# Final Experiment Report", "", "This report is generated from executed artifacts only.", ""]
    for title, path in [
        ("Main Completion", project_root / "tables" / "completion_main.csv"),
        ("Robustness", project_root / "tables" / "robustness.csv"),
        ("Cross Dataset", project_root / "tables" / "cross_dataset.csv"),
        ("Per-Object Risk Analysis", project_root / "tables" / "per_object.csv"),
        ("Pose Refinement Proxy", project_root / "tables" / "pose_refinement.csv"),
        ("Robot Feasibility Proxy", project_root / "tables" / "robot_feasibility_proxy.csv"),
        ("Ablation", project_root / "tables" / "ablation.csv"),
    ]:
        report += [f"## {title}", ""]
        if path.exists():
            rows = read_csv_dicts(path)
            report += [f"- Source: `{path}`", f"- Rows: {len(rows)}"]
            if rows:
                report += ["", "| " + " | ".join(rows[0].keys()) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"]
                for row in rows[:12]:
                    report.append("| " + " | ".join(str(v) for v in row.values()) + " |")
                if len(rows) > 12:
                    report.append(f"\nAdditional rows omitted here: {len(rows) - 12}")
        else:
            report.append("- Not run yet.")
        report.append("")
    viz = project_root / "visualizations" / "samples" / "visualization_index.csv"
    if viz.exists():
        report += ["## Visualizations", "", f"- Source: `{viz}`", f"- Samples: {len(read_csv_dicts(viz))}", ""]
    write_text(project_root / "docs" / "FINAL_EXPERIMENT_REPORT.md", "\n".join(report) + "\n")

    completion_rows = read_csv_dicts(project_root / "tables" / "completion_main.csv") if (project_root / "tables" / "completion_main.csv").exists() else []
    summary = ["# ICRA Result Summary", ""]
    if completion_rows:
        summary += ["## Main Quantitative Result", ""]
        for row in completion_rows:
            summary.append(f"- {row['model']}: CD-L1={row.get('cd_l1_mean', '')}, F-score={row.get('f_score_mean', '')}, n={row.get('samples', '')}")
    else:
        summary.append("No completion metrics have been executed yet.")
    write_text(project_root / "docs" / "ICRA_RESULT_SUMMARY.md", "\n".join(summary) + "\n")

    checklist = [
        "# Final Checklist",
        "",
        f"- [{'x' if (project_root / 'status' / '01_audit_environment.done').exists() else ' '}] Environment audit",
        f"- [{'x' if (project_root / 'status' / '02_audit_repositories.done').exists() else ' '}] Repository audit",
        f"- [{'x' if (project_root / 'status' / '05_build_dataset_contracts.done').exists() else ' '}] Dataset contracts",
        f"- [{'x' if (project_root / 'status' / '17_train_rapc_net_full.done').exists() else ' '}] RAPC-Net training",
        f"- [{'x' if (project_root / 'status' / '18_evaluate_completion_full.done').exists() else ' '}] Completion evaluation",
        f"- [{'x' if (project_root / 'status' / '20_evaluate_robustness_full.done').exists() else ' '}] Robustness evaluation",
        f"- [{'x' if (project_root / 'status' / '22_evaluate_robot_proxy.done').exists() else ' '}] Robot feasibility proxy",
        f"- [{'x' if (project_root / 'status' / '23_run_ablations_full.done').exists() else ' '}] Ablation evaluation",
    ]
    write_text(project_root / "docs" / "FINAL_CHECKLIST.md", "\n".join(checklist) + "\n")
    return checks
