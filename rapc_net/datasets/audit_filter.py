"""Explicit audit membership; never silently treat unaudited pairs as clean."""
import csv
import hashlib
from pathlib import Path


def load_audit(path: Path, split: str):
    records = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            sid = row["sample_id"]
            if sid in records:
                raise ValueError(f"Duplicate audit entry: {split}/{sid}")
            flag = row["grossly_disjoint"].lower()
            if flag not in {"true", "false"}:
                raise ValueError(f"Invalid audit flag for {sid}: {flag}")
            records[sid] = flag == "true"
    return records


def filter_paths(paths, report: Path, split: str):
    records = load_audit(report, split)
    missing = [p.stem for p in paths if p.stem not in records]
    if missing:
        raise ValueError(f"Audit does not cover {len(missing)} {split} pairs, e.g. {missing[:3]}; rerun the audit")
    kept = [p for p in paths if not records[p.stem]]
    excluded = [p.stem for p in paths if records[p.stem]]
    return kept, {"audit_path": str(report), "audit_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                  "split": split, "total": len(paths), "kept": len(kept), "excluded": excluded}
