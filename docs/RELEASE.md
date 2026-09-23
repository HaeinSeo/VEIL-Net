# GitHub and Hugging Face Release

For the single-file source-and-weights distribution, use
`scripts/build_public_bundle.py`; the [bundle guide](PUBLIC_BUNDLE_KO.md) documents
installation, evaluation, training, and the target-qualified release gate.
`scripts/prepare_release.py` below remains a **source-only** export.

## Build a Curated GitHub Tree

From the original research workspace:

```bash
python scripts/prepare_release.py --output releases/veil-net-github
```

The allowlist includes public/core source, focused tests, portable configuration,
benchmark CSV/JSON, documentation, license, and CI. It excludes datasets, weights,
predictions, local environments, backup folders, and intermediate paper drafts.
`release_manifest.json` records every staged file's SHA-256.

Run in the staged folder before creating your GitHub repository:

```bash
python -m pip install -e ".[dev,hub]"
python -m veil_net smoke
python -m pytest -q
python -m build
```

This research workspace may be inside a larger Git repository. The staged folder
is intended to become its own repository with `git init`; use its root for the
first commit. Set your actual remote URL after creating the GitHub repository.

## Export a Hugging Face Model

Run in the research workspace where the checkpoint exists:

```bash
python -m veil_net export --checkpoint checkpoints/rapc_coverage_balance/seed_0/last.pt --output releases/veil-model
```

The command verifies exact tensor equality and writes weights, configuration,
model card, and provenance. It does not publish. Model configuration is taken
from the checkpoint rather than inferred from filenames or current defaults.

Use the [Hugging Face upload workflow](https://huggingface.co/docs/huggingface_hub/guides/upload)
when you are ready to publish:

```bash
hf auth login
hf upload YOUR_ACCOUNT/VEIL-Net releases/veil-model . --repo-type model
```

Replace `YOUR_ACCOUNT/VEIL-Net` with your actual model repository. The model
card follows [Hub model-card metadata](https://huggingface.co/docs/hub/model-cards).
The loader uses [Hub file downloads](https://huggingface.co/docs/huggingface_hub/guides/download)
and [safetensors](https://huggingface.co/docs/safetensors/en/api/torch); it does not
execute remote Python code.

## Record the Published Version

Add the actual GitHub and Hub URLs to the README/model card once they exist.
Use a code release tag and a Hub commit revision together. Link the benchmark
protocol only to the checkpoint hash that produced its numbers. The current
external F-score report is kept separately until its raw export is attached.

The original project code has an MIT license. Third-party datasets and baseline
implementations are not included in the release; their original terms apply.
