import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def release_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_release", ROOT / "scripts" / "prepare_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_excludes_heavy_and_local_files(tmp_path):
    release = release_module()
    output = tmp_path / "github"
    report = release.stage_release(ROOT, output)
    manifest = json.loads((output / "release_manifest.json").read_text())
    assert report["files"] == len(manifest["sha256"])
    assert "veil_net/model.py" in manifest["sha256"]
    assert "benchmarks/validation_protocol.json" in manifest["sha256"]
    assert "figures/veil-mascot.png" in manifest["sha256"]
    assert "figures/scene-completion.png" in manifest["sha256"]
    assert "figures/qualitative-comparison.png" in manifest["sha256"]
    assert "docs/RESULTS.md" in manifest["sha256"]
    assert "docs/TRAINING.md" in manifest["sha256"]
    assert "CITATION.cff" in manifest["sha256"]
    assert not any(
        path.startswith(("cache/", "checkpoints/", "backups/", "results/", ".venv"))
        for path in manifest["sha256"]
    )
    assert not any(path.endswith((".pt", ".npz", ".safetensors")) for path in manifest["sha256"])
    with pytest.raises(FileExistsError):
        release.stage_release(ROOT, output)


def test_public_protocol_has_unique_cohort_and_checkpoint_identity():
    protocol = json.loads((ROOT / "benchmarks" / "validation_protocol.json").read_text())
    assert len(protocol["sample_ids"]) == len(set(protocol["sample_ids"])) == 929
    assert protocol["eval_points"] == 4096
    assert len(protocol["checkpoint_sha256"]) == 64
