"""Stage a small GitHub source tree using an explicit file allowlist. No upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = [
    "README.md",
    "README_KO.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
    "MANIFEST.in",
    "CITATION.cff",
]
DOCUMENTS = [
    "ARCHITECTURE.md",
    "DATA_FORMAT.md",
    "RELEASE.md",
    "QUICKSTART_KO.md",
    "PAPER_RESULTS_KO.md",
    "TARGET_FINETUNE_KO.md",
    "TARGET_SEARCH_KO.md",
    "PUBLIC_BUNDLE_KO.md",
    "RESULTS.md",
    "TRAINING.md",
]
TESTS = [
    "test_metrics.py",
    "test_rapc_net.py",
    "test_geometry_queries.py",
    "test_spatial_completion.py",
    "test_sparse_observations.py",
    "test_geometry_fix.py",
    "test_surface_loss.py",
]


def release_files(root: Path) -> list[Path]:
    paths = [root / name for name in TOP_LEVEL]
    paths.extend(root / "docs" / name for name in DOCUMENTS)
    paths.extend(root / "tests" / name for name in TESTS)
    paths.extend((root / "tests").glob("test_public_*.py"))
    for package in ("veil_net", "rapc_net", "adapters"):
        paths.extend((root / package).rglob("*.py"))
    paths.append(root / "veil_net" / "model_card.md")
    paths.extend((root / "configs" / "release").glob("*.yaml"))
    for pattern in ("*.csv", "*.json", "*.md"):
        paths.extend((root / "benchmarks").glob(pattern))
    paths.extend((root / ".github" / "workflows").glob("*.yml"))
    paths.extend((root / "figures").glob("*.png"))
    paths.append(root / "figures" / "README.md")
    paths.extend(
        root / "scripts" / name
        for name in (
            "prepare_release.py",
            "prepare_benchmarks.py",
            "render_paper_results.py",
            "verify_paper_results.py",
            "finetune_target_metrics.py",
            "search_target_metrics.py",
            "build_public_bundle.py",
            "verify_bundle.py",
        )
    )
    result = sorted(set(paths))
    for path in result:
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(f"Expected a regular source file inside the project: {path}")
    return result


def stage_release(root: Path, output: Path) -> dict:
    paths = release_files(root)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new release directory: {output}")
    output.mkdir(parents=True)
    manifest = {}
    for source in paths:
        relative = source.relative_to(root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    report = {"files": len(manifest), "sha256": manifest}
    (output / "release_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return {"output": str(output), "files": len(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_release(ROOT, args.output), indent=2))


if __name__ == "__main__":
    main()
