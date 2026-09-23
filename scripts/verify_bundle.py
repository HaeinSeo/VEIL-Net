"""Check public-bundle file integrity without loading PyTorch or datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["sha256"]
    errors = []
    if manifest["files"] != len(expected):
        errors.append("Manifest file count mismatch")
    for name, checksum in expected.items():
        relative = PurePosixPath(name)
        path = root.joinpath(*relative.parts)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.resolve().is_relative_to(root)
        ):
            errors.append(f"Invalid manifest path: {name}")
        elif path.is_symlink() or not path.is_file():
            errors.append(f"Missing or nonregular file: {name}")
        elif digest(path) != checksum:
            errors.append(f"SHA-256 mismatch: {name}")
    try:
        status = json.loads((root / "PUBLICATION_STATUS.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        status = {}
    if not isinstance(status.get("paper_targets_met"), bool):
        errors.append("Invalid publication status")
    return {
        "integrity_passed": not errors,
        "files_checked": len(expected),
        "paper_targets_met": status.get("paper_targets_met") is True,
        "errors": errors,
        "scope": "Integrity of listed release files, not a fresh model evaluation.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-targets", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.root)
    print(json.dumps(result, indent=2))
    return (
        0
        if result["integrity_passed"] and (not args.require_targets or result["paper_targets_met"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
