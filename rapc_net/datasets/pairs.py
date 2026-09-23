from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rapc_net.utils.io import read_csv_dicts, write_json
from rapc_net.utils.pointcloud import normalize_unit_sphere, sample_points, transform_points, validate_points
from rapc_net.datasets.pair_quality import pair_alignment_report


def _nearest_distances_chunked(src: np.ndarray, dst: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = np.empty((len(src),), dtype=np.float32)
    for start in range(0, len(src), chunk):
        block = src[start : start + chunk]
        diff = block[:, None, :] - dst[None, :, :]
        out[start : start + chunk] = np.sqrt(np.sum(diff * diff, axis=2).min(axis=1))
    return out


def _risk_pseudo_label(partial: np.ndarray, complete: np.ndarray, seed: int) -> np.ndarray:
    partial = validate_points(partial, "partial")
    complete = validate_points(complete, "complete")
    gt_ref = sample_points(complete, min(len(complete), 4096), seed + 104729)
    cad_dist = _nearest_distances_chunked(partial, gt_ref)
    noise = np.clip(cad_dist / (np.quantile(cad_dist, 0.95) + 1e-6), 0.0, 1.0)

    diff = partial[:, None, :] - partial[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    k = min(12, max(2, len(partial) - 1))
    kth = np.partition(dist, k, axis=1)[:, k]
    sparse = np.clip((kth - np.quantile(kth, 0.15)) / (np.quantile(kth, 0.95) - np.quantile(kth, 0.15) + 1e-6), 0.0, 1.0)

    center = partial.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(partial - center, axis=1)
    boundary = np.clip((radius - np.quantile(radius, 0.55)) / (np.quantile(radius, 0.97) - np.quantile(radius, 0.55) + 1e-6), 0.0, 1.0)

    risk = 0.45 * boundary + 0.35 * sparse + 0.20 * noise
    return np.clip(risk, 0.0, 1.0).astype(np.float32)


def load_npz_pair(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        partial = validate_points(data["partial"], "partial")
        complete = validate_points(data["complete"], "complete") if "complete" in data.files else None
        return {"partial": partial, "complete": complete, "path": str(path)}


def save_pair(path: Path, partial: np.ndarray, complete: np.ndarray | None, metadata: dict[str, Any]) -> None:
    if complete is not None:
        report = pair_alignment_report(partial, complete)
        if report["grossly_disjoint"]:
            raise ValueError(f"Gross input/GT separation; check depth/mask/pose correspondence: {report}")
    path.parent.mkdir(parents=True, exist_ok=True)
    norm_partial, transform = normalize_unit_sphere(partial)
    arrays: dict[str, Any] = {"partial": norm_partial.astype(np.float32)}
    if complete is not None:
        sampled_complete = sample_points(complete, int(metadata.get("output_points", 16384)), int(metadata.get("seed", 0)))
        center = np.asarray(transform["center"], dtype=np.float32)
        scale = float(transform["scale"])
        norm_complete = ((sampled_complete - center) / scale).astype(np.float32)
        arrays["complete"] = norm_complete
        arrays["risk_target"] = _risk_pseudo_label(arrays["partial"], norm_complete, int(metadata.get("seed", 0)))
    np.savez_compressed(path, **arrays)
    write_json(path.with_suffix(".json"), metadata | {"normalization": transform})


def _read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def _load_bop_camera(path: Path, frame_id: str) -> tuple[np.ndarray, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    item = data[str(int(frame_id))]
    k = np.asarray(item["cam_K"], dtype=np.float32).reshape(3, 3)
    depth_scale = float(item.get("depth_scale", 1.0))
    return k, depth_scale


def _load_bop_pose(path: Path, frame_id: str, object_index: str) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    obj = data[str(int(frame_id))][int(object_index)]
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.asarray(obj["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    mat[:3, 3] = np.asarray(obj["cam_t_m2c"], dtype=np.float32) / 1000.0
    return mat


def _depth_mask_to_points_m(depth_path: Path, mask_path: Path, camera_path: Path, frame_id: str) -> np.ndarray:
    depth = _read_image(depth_path).astype(np.float32)
    mask = _read_image(mask_path) > 0
    if depth.shape[:2] != mask.shape[:2]:
        raise ValueError(f"depth/mask shape mismatch: {depth_path} {depth.shape}, {mask_path} {mask.shape}")
    k, depth_scale = _load_bop_camera(camera_path, frame_id)
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if len(xs) == 0:
        raise ValueError(f"mask selects no valid depth pixels: {mask_path}")
    z = depth[ys, xs] * depth_scale / 1000.0
    x = (xs.astype(np.float32) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float32) - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _read_ply_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"invalid PLY without end_header: {path}")
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        fmt = next((line for line in header_lines if line.startswith("format ")), "")
        vertex_count = 0
        face_count = 0
        vertex_props: list[str] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if parts[1] == "vertex":
                    vertex_count = int(parts[2])
                elif parts[1] == "face":
                    face_count = int(parts[2])
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                vertex_props.append(parts[-1])
        if "ascii" in fmt:
            vertices = []
            for _ in range(vertex_count):
                vals = f.readline().decode("ascii", errors="replace").split()
                vertices.append([float(vals[vertex_props.index(axis)]) for axis in ("x", "y", "z")])
            faces = []
            for _ in range(face_count):
                vals = f.readline().decode("ascii", errors="replace").split()
                n = int(vals[0])
                if n >= 3:
                    faces.append([int(vals[1]), int(vals[2]), int(vals[3])])
            return np.asarray(vertices, dtype=np.float32) / 1000.0, np.asarray(faces, dtype=np.int32)
        if "binary_little_endian" not in fmt:
            raise ValueError(f"unsupported PLY format in {path}: {fmt}")
        type_map = {"float": "f", "float32": "f", "double": "d", "uchar": "B", "uint8": "B", "char": "b", "int": "i", "int32": "i", "uint": "I", "uint32": "I"}
        prop_types = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                prop_types.append((parts[1], parts[-1]))
        fmt_chars = "<" + "".join(type_map[t] for t, _ in prop_types)
        size = struct.calcsize(fmt_chars)
        vertices = []
        for _ in range(vertex_count):
            vals = struct.unpack(fmt_chars, f.read(size))
            by_name = {name: vals[i] for i, (_, name) in enumerate(prop_types)}
            vertices.append([by_name["x"], by_name["y"], by_name["z"]])
        return np.asarray(vertices, dtype=np.float32) / 1000.0, np.empty((0, 3), dtype=np.int32)


def _sample_mesh(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    vertices = validate_points(vertices, "vertices")
    if len(faces) == 0:
        return sample_points(vertices, count, seed)
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    valid = areas > 0
    if not valid.any():
        return sample_points(vertices, count, seed)
    tri = tri[valid]
    probs = areas[valid] / areas[valid].sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tri), size=count, p=probs)
    chosen = tri[idx]
    u = rng.random((count, 1), dtype=np.float32)
    v = rng.random((count, 1), dtype=np.float32)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0]) + v * (chosen[:, 2] - chosen[:, 0])


def _prepare_bop_row(project_root: Path, split: str, row: dict[str, str], seed: int, input_points: int, output_points: int) -> Path:
    required = ["depth_path", "mask_path", "camera_path", "pose_path", "cad_path", "frame_id", "object_index", "sample_id"]
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(f"row {row.get('sample_id')} missing required fields: {missing}")
    partial = _depth_mask_to_points_m(Path(row["depth_path"]), Path(row["mask_path"]), Path(row["camera_path"]), row["frame_id"])
    partial = sample_points(partial, input_points, seed)
    vertices, faces = _read_ply_vertices_faces(Path(row["cad_path"]))
    cad_points = _sample_mesh(vertices, faces, output_points, seed)
    pose = _load_bop_pose(Path(row["pose_path"]), row["frame_id"], row["object_index"])
    complete = transform_points(cad_points, pose)
    dest = project_root / "cache" / "completion_pairs" / split / row["dataset"] / f"{row['sample_id']}.npz"
    save_pair(
        dest,
        partial,
        complete,
        {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "object_id": row["object_id"],
            "scene_id": row["scene_id"],
            "frame_id": row["frame_id"],
            "pose_path": row["pose_path"],
            "camera_path": row["camera_path"],
            "cad_path": row["cad_path"],
            "input_points": input_points,
            "output_points": output_points,
            "seed": seed,
            "coordinate_unit": "meter",
        },
    )
    return dest


def prepare_pairs_from_manifest(project_root: Path, split: str, max_samples: int | None, seed: int) -> int:
    manifest = project_root / "manifests" / f"{split}.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is missing; run 06_build_manifests.py first")
    rows = read_csv_dicts(manifest)
    if max_samples is not None:
        rows = rows[:max_samples]
    made = 0
    skipped: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("complete_gt_allowed", "")).lower() != "true":
            skipped.append({"sample_id": row.get("sample_id", ""), "reason": "complete_gt_allowed is false"})
            continue
        try:
            _prepare_bop_row(project_root, split, row, seed, input_points=2048, output_points=16384)
            made += 1
        except Exception as exc:
            skipped.append({"sample_id": row.get("sample_id", ""), "reason": str(exc)})
    write_json(project_root / "results" / f"completion_pair_skips_{split}.json", skipped)
    if made == 0 and rows:
        reasons = "; ".join(sorted({item["reason"] for item in skipped})[:5])
        raise RuntimeError(f"No completion pairs were generated from {len(rows)} rows. Main reasons: {reasons}")
    return made
