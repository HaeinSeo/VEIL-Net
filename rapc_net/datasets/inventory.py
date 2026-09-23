from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rapc_net.utils.io import write_csv_dicts, write_json, write_text, write_yaml_like


KNOWN_DATASETS = {
    "graspnet": ["graspnet"],
    "hb": ["hb", "homebreweddb"],
    "tless": ["t-less", "tless", "less"],
    "lmo": ["lm-o", "lmo", "linemod_occlusion"],
    "ocid": ["ocid"],
    "tudl": ["tud-l", "tudl"],
    "ycb_video": ["ycb", "ycb-video", "ycb_video"],
    "uwis_occluded": ["uwis", "uwis-occluded"],
}

RGB_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DEPTH_HINTS = {"depth", "depths"}
MASK_HINTS = {"mask", "masks", "label", "seg", "instance"}
CAD_EXT = {".ply", ".obj", ".stl", ".off"}


def _name_match(path: Path) -> str | None:
    lowered = path.name.lower()
    for canonical, aliases in KNOWN_DATASETS.items():
        if any(alias in lowered for alias in aliases):
            return canonical
    return None


def discover_dataset_roots(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    roots: list[Path] = []
    for child in data_root.iterdir():
        if child.is_dir() and (_name_match(child) or (child / "scene_gt.json").exists() or (child / "models").exists()):
            roots.append(child)
    return sorted(roots)


def inspect_dataset(root: Path, precise: bool = False, max_files: int | None = None) -> dict[str, Any]:
    counts = {
        "rgb_files": 0,
        "depth_files": 0,
        "mask_files": 0,
        "cad_files": 0,
        "pose_files": 0,
        "camera_files": 0,
        "empty_files": 0,
    }
    samples: list[str] = []
    digest = hashlib.sha256()
    visited = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        visited += 1
        if max_files is not None and visited > max_files:
            break
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        lower_parts = {p.lower() for p in path.parts}
        if suffix in RGB_EXT and not (lower_parts & DEPTH_HINTS) and not (lower_parts & MASK_HINTS):
            counts["rgb_files"] += 1
        if (lower_parts & DEPTH_HINTS) or "depth" in path.stem.lower():
            counts["depth_files"] += 1
        if (lower_parts & MASK_HINTS) or "mask" in path.stem.lower():
            counts["mask_files"] += 1
        if suffix in CAD_EXT or "models" in lower_parts:
            counts["cad_files"] += int(suffix in CAD_EXT)
        if path.name.lower() in {"scene_gt.json", "poses.txt"} or "pose" in path.stem.lower():
            counts["pose_files"] += 1
        if "camera" in path.stem.lower() or "intrinsic" in path.stem.lower() or path.name.lower() == "scene_camera.json":
            counts["camera_files"] += 1
        try:
            if path.stat().st_size == 0:
                counts["empty_files"] += 1
            if precise:
                digest.update(rel.encode("utf-8"))
                digest.update(str(path.stat().st_size).encode("ascii"))
        except OSError:
            counts["empty_files"] += 1
        if len(samples) < 10:
            samples.append(rel)
    canonical = _name_match(root) or root.name.lower().replace("-", "_")
    return {
        "dataset": canonical,
        "path": str(root),
        "visited_files": visited,
        "truncated": max_files is not None and visited > max_files,
        "has_rgb": counts["rgb_files"] > 0,
        "has_depth": counts["depth_files"] > 0,
        "has_mask": counts["mask_files"] > 0,
        "has_camera": counts["camera_files"] > 0,
        "has_pose": counts["pose_files"] > 0,
        "has_cad": counts["cad_files"] > 0,
        "counts": counts,
        "sample_files": samples,
        "fast_hash": digest.hexdigest() if precise else "",
    }


def write_inventory_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for item in rows:
        flat.append(
            {
                "dataset": item["dataset"],
                "path": item["path"],
                "visited_files": item["visited_files"],
                "has_rgb": item["has_rgb"],
                "has_depth": item["has_depth"],
                "has_mask": item["has_mask"],
                "has_camera": item["has_camera"],
                "has_pose": item["has_pose"],
                "has_cad": item["has_cad"],
                "empty_files": item["counts"]["empty_files"],
            }
        )
    write_csv_dicts(root / "manifests" / "dataset_inventory.csv", flat)
    write_json(root / "results" / "dataset_inventory.json", rows)
    lines = ["# Dataset Inventory", ""]
    if not rows:
        lines.append("No dataset roots were discovered under the configured data root.")
    for item in rows:
        lines += [
            f"## {item['dataset']}",
            f"- Path: `{item['path']}`",
            f"- RGB/depth/mask/camera/pose/CAD: {item['has_rgb']}/{item['has_depth']}/{item['has_mask']}/{item['has_camera']}/{item['has_pose']}/{item['has_cad']}",
            f"- Visited files: {item['visited_files']}",
            "",
        ]
    write_text(root / "docs" / "DATASET_INVENTORY.md", "\n".join(lines))


def build_contracts(project_root: Path, inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contracts = []
    for item in inventory:
        completion = bool(item["has_depth"] and item["has_mask"] and item["has_pose"] and item["has_cad"])
        pose = bool(item["has_depth"] and item["has_camera"] and item["has_pose"] and item["has_cad"])
        robustness = bool(item["has_depth"])
        qualitative = bool(item["has_rgb"] or item["has_depth"])
        contract = {
            "dataset": item["dataset"],
            "root": item["path"],
            "completion_quantitative": completion,
            "pose_refinement": pose,
            "fusion": pose and item["has_rgb"],
            "ood_robustness": robustness and not completion,
            "qualitative": qualitative,
            "complete_gt_allowed": completion,
            "reason": "CAD and pose available" if completion else "Missing CAD or valid pose; Chamfer target disabled",
        }
        contracts.append(contract)
        write_yaml_like(project_root / "configs" / "datasets" / f"{contract['dataset']}.yaml", contract)
    lines = ["# Dataset Contracts", ""]
    for c in contracts:
        lines += [
            f"## {c['dataset']}",
            f"- Completion quantitative: {c['completion_quantitative']}",
            f"- Pose refinement: {c['pose_refinement']}",
            f"- Complete GT allowed: {c['complete_gt_allowed']}",
            f"- Decision: {c['reason']}",
            "",
        ]
    write_text(project_root / "docs" / "DATASET_CONTRACTS.md", "\n".join(lines))
    return contracts
