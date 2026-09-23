from __future__ import annotations

import json
from pathlib import Path

from rapc_net.utils.io import read_json, write_csv_dicts, write_text


def _find_model_dir(dataset_root: Path) -> Path | None:
    candidates = [
        dataset_root / "models",
        dataset_root / "models_eval",
        dataset_root / "ycbv_models" / "models",
        dataset_root / "tudl_models" / "models",
        dataset_root / "tudl_models" / "models_eval",
        dataset_root / "HB" / "hb_models" / "models",
        dataset_root / "HB" / "hb_models" / "models_eval",
        dataset_root / "tless" / "tless_models" / "models_eval",
        dataset_root / "tless" / "tless_models" / "models_cad",
    ]
    for current in candidates:
        if current.exists() and any(current.glob("obj_*.ply")):
            return current
    return None


def _find_bop_scenes(dataset_root: Path) -> list[Path]:
    roots = [
        dataset_root / "test",
        dataset_root / "train",
        dataset_root / "ycbv_test_all" / "test",
        dataset_root / "ycbv_test_bop19" / "test",
        dataset_root / "tudl_test_all" / "test",
        dataset_root / "tudl_test_bop19" / "test",
        dataset_root / "HB" / "hb_test_kinect_all" / "test_kinect",
        dataset_root / "HB" / "hb_test_primesense_all" / "test_primesense",
        dataset_root / "HB" / "hb_test_primesense_bop19" / "test_primesense",
        dataset_root / "tless" / "tless_test_primesense_all" / "test_primesense",
        dataset_root / "tless" / "tless_test_primesense_bop19" / "test_primesense",
        dataset_root / "tless" / "test_primesense",
    ]
    scenes: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and (child / "scene_gt.json").exists() and (child / "scene_camera.json").exists():
                scenes.append(child)
    return sorted(set(scenes))


def _model_dir_for_scene(dataset_root: Path, scene: Path) -> tuple[str, Path | None]:
    scene_text = scene.as_posix().lower()
    if "/tless/" in scene_text or "\\tless\\" in str(scene).lower():
        candidates = [dataset_root / "tless" / "tless_models" / "models_eval", dataset_root / "tless" / "tless_models" / "models_cad"]
        dataset = "tless"
    elif "/hb/" in scene_text or "\\hb\\" in str(scene).lower():
        candidates = [dataset_root / "HB" / "hb_models" / "models", dataset_root / "HB" / "hb_models" / "models_eval"]
        dataset = "hb"
    else:
        candidates = [_find_model_dir(dataset_root)]
        dataset = "ycb_video" if "ycb" in dataset_root.name.lower() else dataset_root.name.lower().replace("-", "_")
    for candidate in candidates:
        if candidate is not None and candidate.exists() and any(candidate.glob("obj_*.ply")):
            return dataset, candidate
    return dataset, None


def _image_path(scene: Path, folder: str, frame_id: int) -> Path | None:
    base = scene / folder
    if not base.exists():
        return None
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = base / f"{frame_id:06d}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _bop_rows(dataset: str, dataset_root: Path, max_rows: int | None = None) -> list[dict[str, str | bool]]:
    rows: list[dict[str, str | bool]] = []
    scenes = _find_bop_scenes(dataset_root)
    for scene in scenes:
        row_dataset, model_dir = _model_dir_for_scene(dataset_root, scene)
        try:
            gt = json.loads((scene / "scene_gt.json").read_text(encoding="utf-8"))
            camera = json.loads((scene / "scene_camera.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        scene_id = scene.name
        for frame_key, objects in gt.items():
            frame_id = int(frame_key)
            depth = _image_path(scene, "depth", frame_id)
            rgb = _image_path(scene, "rgb", frame_id)
            cam_ok = frame_key in camera
            if depth is None or not cam_ok:
                continue
            for obj_index, obj in enumerate(objects):
                obj_id = int(obj["obj_id"])
                mask = scene / "mask_visib" / f"{frame_id:06d}_{obj_index:06d}.png"
                if not mask.exists():
                    mask = scene / "mask" / f"{frame_id:06d}_{obj_index:06d}.png"
                cad = model_dir / f"obj_{obj_id:06d}.ply" if model_dir is not None else None
                complete_ok = bool(mask.exists() and cad is not None and cad.exists())
                rows.append(
                    {
                        "sample_id": f"{row_dataset}_{scene_id}_{frame_id:06d}_{obj_index:02d}",
                        "dataset": row_dataset,
                        "object_id": str(obj_id),
                        "scene_id": scene_id,
                        "frame_id": f"{frame_id:06d}",
                        "depth_path": str(depth),
                        "mask_path": str(mask) if mask.exists() else "",
                        "rgb_path": str(rgb) if rgb else "",
                        "cad_path": str(cad) if cad is not None and cad.exists() else "",
                        "pose_path": str(scene / "scene_gt.json"),
                        "camera_path": str(scene / "scene_camera.json"),
                        "object_index": str(obj_index),
                        "complete_gt_allowed": complete_ok,
                    }
                )
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
    return rows


def _split_rows(rows: list[dict[str, str | bool]], seed: int, split_policy: str) -> dict[str, list[dict[str, str | bool]]]:
    import random

    rng = random.Random(seed)
    if split_policy == "row":
        shuffled = rows[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        return {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}

    if split_policy not in {"scene", "object"}:
        raise ValueError("--split-policy must be one of row, scene, object")
    groups: dict[str, list[dict[str, str | bool]]] = {}
    for row in rows:
        if split_policy == "object":
            key = f"{row.get('dataset', '')}:{row.get('object_id', '')}"
        else:
            key = f"{row.get('dataset', '')}:{row.get('scene_id', '')}"
        groups.setdefault(key, []).append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    split_keys = {
        "train": set(keys[:train_end]),
        "val": set(keys[train_end:val_end]),
        "test": set(keys[val_end:]),
    }
    return {name: [row for key in keys if key in allowed for row in groups[key]] for name, allowed in split_keys.items()}


def _overlap_report(splits: dict[str, list[dict[str, str | bool]]], field: str) -> dict[str, list[str]]:
    sets = {name: {str(row.get(field, "")) for row in rows if row.get(field, "")} for name, rows in splits.items()}
    return {
        "train_val": sorted(sets["train"] & sets["val"])[:20],
        "train_test": sorted(sets["train"] & sets["test"])[:20],
        "val_test": sorted(sets["val"] & sets["test"])[:20],
    }


def build_splits(project_root: Path, seed: int = 0, max_rows: int | None = None, max_non_bop_per_dataset: int = 0, split_policy: str = "scene") -> dict[str, int]:
    inventory_path = project_root / "results" / "dataset_inventory.json"
    if not inventory_path.exists():
        raise FileNotFoundError("results/dataset_inventory.json is missing; run 04_inventory_datasets.py first")
    inventory = read_json(inventory_path)
    rows = []
    bop_dataset_names = {"hb", "tudl", "ycb_video", "lmo", "tless"}
    for item in inventory:
        if not (item.get("has_depth") and item.get("has_mask")):
            continue
        dataset_root = Path(item["path"])
        remaining = None if max_rows is None else max(max_rows - len(rows), 0)
        bop = _bop_rows(item["dataset"], dataset_root, max_rows=remaining) if item["dataset"] in bop_dataset_names and remaining != 0 else []
        if bop:
            rows.extend(bop)
            if max_rows is not None and len(rows) >= max_rows:
                rows = rows[:max_rows]
                break
            continue
        if max_non_bop_per_dataset <= 0:
            continue
        candidates = []
        stack = [dataset_root]
        while stack and len(candidates) < max_non_bop_per_dataset:
            current = stack.pop()
            try:
                for child in current.iterdir():
                    if child.is_dir():
                        stack.append(child)
                    elif "depth" in child.as_posix().lower():
                        candidates.append(child)
                        if len(candidates) >= max_non_bop_per_dataset:
                            break
            except OSError:
                continue
        for idx, depth in enumerate(candidates):
            rows.append(
                {
                    "sample_id": f"{item['dataset']}_{idx:08d}",
                    "dataset": item["dataset"],
                    "object_id": "unknown",
                    "scene_id": "unknown",
                    "frame_id": depth.stem,
                    "depth_path": str(depth),
                    "mask_path": "",
                    "rgb_path": "",
                    "cad_path": "",
                    "pose_path": "",
                    "camera_path": "",
                    "object_index": "",
                    "complete_gt_allowed": False,
                }
            )
        if max_rows is not None and len(rows) >= max_rows:
            rows = rows[:max_rows]
            break
    n = len(rows)
    main_splits = _split_rows(rows, seed, split_policy)
    splits = main_splits | {"ood_test": [r for r in rows if r["dataset"] in {"ocid", "uwis_occluded"}]}
    for name, split_rows in splits.items():
        write_csv_dicts(project_root / "manifests" / f"{name}.csv", split_rows)
    write_text(
        project_root / "docs" / "SPLIT_REPORT.md",
        "# Split Report\n\n"
        f"- Total candidate rows: {n}\n"
        f"- Split policy: {split_policy}\n"
        f"- Train/val/test/OOD: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}/{len(splits['ood_test'])}\n"
        f"- Scene overlap train/test examples: {_overlap_report(main_splits, 'scene_id')['train_test']}\n"
        f"- Object overlap train/test examples: {_overlap_report(main_splits, 'object_id')['train_test']}\n"
        "- Leakage policy: use `scene` for low leakage, `object` for unseen-object evaluation, or `row` only for quick debugging.\n",
    )
    return {k: len(v) for k, v in splits.items()}
