# Training and evaluation

## Prepare data

Training consumes object-level `.npz` pairs with `partial` and `complete` arrays.
Keep coordinates in their original frame and units; the model handles its own
input-frame normalization. See [the data contract](DATA_FORMAT.md) for keys,
sampling, and split requirements. Datasets are not included in this repository.

The fixed reference cohort contains 929 validation observations. Its IDs are
listed in `benchmarks/validation_protocol.json`. Do not include these observations
in training. Training and validation paths can live outside the repository.

## Train from scratch

```bash
python -m veil_net train --config configs/release/veil_geometry.yaml --pairs cache/completion_pairs/train --device cuda
```

This 50-epoch recipe needs no pretrained checkpoint. It is a reproducible entry
point to the implementation, not a replay of the reference model's staged
65-epoch training. Use a new `run_name` in a copied configuration for a separate
experiment. Add `--resume` only when continuing that same run.

## Export a checkpoint

```bash
python -m veil_net export --checkpoint checkpoints/veil_geometry/seed_0/last.pt --output model
```

Use the checkpoint path produced by your run. Export writes safetensors weights,
the model configuration, a model card, and provenance. The exporter checks exact
tensor equality with the checkpoint. Keep a new output directory for each model.

## Evaluate a trained model

```bash
python -m veil_net evaluate --model model --pairs cache/completion_pairs/val --protocol benchmarks/validation_protocol.json --max-samples 0 --eval-points 4096 --device cpu --output results/full-validation
```

The evaluation directory contains `samples.csv` and `report.json`. The report
records data and model hashes, point count, seed, and the IDs actually evaluated.
Keep these files with the checkpoint when reporting a result.

To check reproduction of the reference checkpoint:

```bash
python scripts/verify_paper_results.py --report results/full-validation/report.json --reference saved
```

To check whether every external target is met or exceeded:

```bash
python scripts/verify_paper_results.py --report results/full-validation/report.json --reference paper --mode target
```

The latter requires all five metrics on the full cohort. It returns a nonzero
exit code when a target is missed; it does not alter any measured value.

## Fine-tune toward the target metrics

The search script requires the reference warm-start checkpoint, including its
training-ID manifest. That file is **not included in the Git repository**.
The source-and-weights bundle places it at `weights/initial.pt`; its SHA-256 is
recorded in `configs/release/veil_metric_finetune.yaml`. Do not substitute arbitrary
weights or remove the identity check.

With the matching initializer and data available, inspect the plan first:

```bash
python scripts/search_target_metrics.py --run-name veil_search_v1 --train-pairs cache/completion_pairs/train --val-pairs cache/completion_pairs/val --max-trials 6 --epochs-per-trial 20 --eval-every 5 --plan-only
```

Remove `--plan-only` and add `--device cuda --eval-device cpu` to start. The default
budget tries three loss profiles with two seeds, up to 20 additional epochs each.
Full validation runs every five epochs. Training includes a differentiable
F-score surrogate at the evaluation thresholds alongside the geometry losses.

Outputs are stored in `results/target_search/veil_search_v1/`:

| File | Meaning |
| --- | --- |
| `best_candidate.pt` | Closest measured candidate, including the initial reference |
| `best_candidate_report.json` | That candidate's full evaluation |
| `evaluations.json` | Metrics at each evaluated training checkpoint |
| `winner.pt` | Created only when all five targets are met |
| `summary.json` | Final target status and output paths |

Resume an interrupted search with the same arguments plus `--resume`. Exit code
2 means the search budget finished without meeting all targets, not that a winner
was found. Selection uses validation results; a separate test evaluation is needed
to report held-out test performance.

## Publishing weights

Publish the exported `model/` directory with its actual evaluation report and
checkpoint identity. Model weights, prepared data, and experiment outputs are
excluded by `.gitignore`. See [release instructions](RELEASE.md) for bundle
creation and Hugging Face publication.
