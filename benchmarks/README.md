# Benchmark Evidence

## Complete Manuscript Results

<!-- BEGIN PAPER RESULTS -->

Complete results from the author-supplied manuscript. The main VEIL-Net row is an **author-reported external run**; coordinate conditioning uses **50-epoch controls**; dataset-wise results and bootstrap intervals describe the **saved checkpoint**.

### Evaluation Cohort

| Dataset | Samples |
| --- | --- |
| T-LESS | 513 |
| TUD-L | 50 |
| YCB-Video | 366 |
| Total | 929 |

### Overall Completion

| Method | CD-L1 ↓ | CD-L2 ↓ | F@0.03 ↑ | F@0.05 ↑ | Dim. MAE ↓ |
| --- | --- | --- | --- | --- | --- |
| Fusion | 0.53061 | 9.53184 | 0.38348 | 0.47510 | 0.34924 |
| PoinTr | 0.55067 | 9.52597 | 0.27878 | 0.39670 | 0.43411 |
| AdaPoinTr | 0.52486 | 9.62343 | 0.32947 | 0.44170 | 0.42771 |
| SnowflakeNet | 0.51613 | 9.50329 | 0.33311 | 0.46123 | 0.43646 |
| SnowflakeNet FT | 0.42271 | 8.34760 | 0.65293 | 0.75463 | 0.34238 |
| **VEIL-Net (external)** | **0.38629** | **6.97737** | **0.72843** | **0.89253** | **0.22846** |

### Improvement over SnowflakeNet FT

| Metric | Improvement |
| --- | --- |
| CD-L1 | **↓ 8.62%** |
| CD-L2 | **↓ 16.41%** |
| F@0.03 | **↑ 7.55%p** |
| F@0.05 | **↑ 13.79%p** |
| Dimension MAE | **↓ 33.27%** |

### Coordinate-Conditioning Ablation

| Setting | CD-L2 ↓ | F@0.03 ↑ | F@0.05 ↑ | Missing CD ↓ | Observed Error ↓ |
| --- | --- | --- | --- | --- | --- |
| Without coordinate conditioning | 7.48764 | 0.61364 | 0.73557 | 0.15204 | 0.01616 |
| Full VEIL-Net (50 epochs) | 7.04963 | 0.63469 | 0.74313 | 0.13964 | 0.01531 |

Coordinate conditioning reduces **Missing CD by 8.15%** and **CD-L2 by 5.85%**, and increases **F@0.03 by 2.11 percentage points** against its removal control.

### Dataset-Wise Analysis

| Dataset | Samples | Method | CD-L1 ↓ | F@0.03 ↑ | Missing CD ↓ | Dim. MAE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| T-LESS | 513 | SnowflakeNet FT | 0.74103 | 0.45083 | 0.23467 | 0.49639 |
| T-LESS | 513 | VEIL-Net | 0.67720 | 0.45357 | 0.25980 | 0.37363 |
| TUD-L | 50 | SnowflakeNet FT | 0.02462 | 0.90172 | 0.01356 | 0.31718 |
| TUD-L | 50 | VEIL-Net | 0.02208 | 0.89986 | 0.01169 | 0.03166 |
| YCB-Video | 366 | SnowflakeNet FT | 0.03092 | 0.90221 | 0.01839 | 0.12997 |
| YCB-Video | 366 | VEIL-Net | 0.02828 | 0.91260 | 0.01551 | 0.05186 |

CD-L1 and Dimension MAE improve on all three datasets. On YCB-Video, all four listed metrics improve.

### Statistical Analysis

| Metric | Mean Difference | 95% CI |
| --- | --- | --- |
| CD-L1 | -0.03642 | [-0.11561, 0.01929] |
| CD-L2 | -1.37024 | [-4.47204, 0.45013] |
| F@0.03 | 0.00551 | [0.00101, 0.01031] |
| Dimension MAE | -0.11393 | [-0.19867, -0.06138] |

Differences are saved VEIL-Net minus SnowflakeNet FT. Paired scene-level bootstrap intervals exclude zero for F@0.03 and Dimension MAE. The dataset-weighted F@0.03 and bootstrap mean difference correspond to the saved score **0.65844**, not the external score **0.72843**.

<!-- END PAPER RESULTS -->

The manuscript tables are preserved at their reported precision in
`paper_main.csv`, `paper_improvements.csv`, `paper_coordinate_ablation.csv`,
`paper_dataset_wise.csv`, and `paper_bootstrap.csv`. `dataset_counts.csv` records
the cohort. The attachment hash and each table's run scope are in `paper_sources.json`.
Full-precision saved analyses are in [dataset_wise.csv](dataset_wise.csv) and
[paired_scene_bootstrap.csv](paired_scene_bootstrap.csv).

## Saved Validation Run

The saved-checkpoint comparison uses the coverage-balance checkpoint
(epoch index 64, 65 completed epochs) and task-finetuned SnowflakeNet on the
same 929 observations: 513 T-LESS, 50 TUD-L, and 366 YCB-Video.

| Model | CD-L1 ↓ | CD-L2 ↓ | F@0.03 ↑ | F@0.05 ↑ | Dim. MAE ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SnowflakeNet FT | 0.42271 | 8.34760 | 0.65293 | **0.75463** | 0.34238 |
| **VEIL-Net** | **0.38629** | **6.97737** | **0.65844** | 0.75428 | **0.22846** |

Relative reductions: **8.62% CD-L1**, **16.41% CD-L2**, **33.27% Dimension MAE**.
F@0.03 increases by 0.55 percentage points. Missing-region distance is
0.15020 vs. 0.13756 and observed-region error is 0.01612 vs. 0.01463.
All seven metrics are included in [validation.csv](validation.csv).

## Equal-Budget Component Study

Separately trained 50-epoch controls, each evaluated on the same 929 observations.
Bold indicates the best value within this table.

| Setting | CD-L2 ↓ | F@0.03 ↑ | Missing CD ↓ | Observed Error ↓ |
| :--- | ---: | ---: | ---: | ---: |
| Full geometry-query model | 7.04963 | 0.63469 | 0.13964 | **0.01531** |
| Without query coordinates | 7.48764 | 0.61364 | 0.15204 | 0.01616 |
| Without observation-quality signal | 7.70121 | 0.61013 | 0.14050 | 0.01543 |
| Without observed-surface preservation | **6.68220** | **0.63566** | **0.13941** | 0.03166 |

Query-coordinate conditioning reduces missing-region distance by **8.15%**
relative to its removal control. Observed-surface preservation reduces observed
error by **51.66%** relative to its removal control. The controls expose each
component's contribution and the associated geometric trade-offs.
Complete metrics, including CD-L1 and Dimension MAE, are in [ablation.csv](ablation.csv).
The full 50-epoch ablation model is distinct from the 65-epoch headline checkpoint.

## Protocol

- Prepared input / predicted surface: 2,048 / 16,384 points.
- Metric evaluation: independently sample at most 4,096 points per cloud with NumPy seed 0 for prediction, 1 for GT, and 2 for observed points.
- Aggregate: mean of per-observation metrics, with no macro-average by dataset.
- Frame: original prepared-pair coordinates; no GT alignment at inference.
- Split: validation observations after training-ID exclusion; IDs are frozen in [validation_protocol.json](validation_protocol.json).
- Checkpoint SHA-256: `1e7cac6daed3f5ffab7a187fe5dd191d549de0f95525d7e870a4a0d5715a215d`.

CD-L1 is the **sum** of both directed mean Euclidean nearest-neighbor distances.
CD-L2 is the sum of both directed mean **squared** distances. There is no extra
display multiplier. F-score is the harmonic mean of precision and recall with
strict distance `< threshold`; values range from 0 to 1. Thresholds use pair
coordinate units, not automatically millimeters. Dimension MAE is the mean
absolute error of the three axis-aligned spans of the evaluated samples.

Missing CD is the directed mean distance from GT points farther than 0.01 from
the observed cloud to the prediction. Observed Error is the directed mean
distance from observed points to the prediction. The historical name "Missing
CD" refers to this one-directional coverage metric.

The checkpoint and validation protocol support geometric completion comparisons
on this cohort. This is a development validation result, not an unseen-object
benchmark or a downstream grasp-success measurement.

## External Report

The author additionally reported F@0.03 = 0.72843 and F@0.05 = 0.89253 from an
external run, alongside the same rounded CD and dimension values. These are
preserved in [external_reported.csv](external_reported.csv). Its raw predictions,
checkpoint identity, and complete evaluation protocol are not in this workspace;
it is a separately attributed result rather than the saved checkpoint's score.

## Reproduction

```bash
python -m veil_net evaluate --model releases/veil-model --pairs cache/completion_pairs/val --protocol benchmarks/validation_protocol.json --max-samples 0 --eval-points 4096 --output results/reproduced-validation
```

Use `--max-samples 16 --eval-points 1024` for a quick execution check. Changing
the cohort or resolution produces a different evaluation, recorded in the JSON
report. Prepared data must be obtained separately; it is not redistributed here.

`sources.json` records the original CSV/protocol paths and hashes. In the original
research workspace, `python scripts/prepare_benchmarks.py` rebuilds the public
CSV files from those frozen outputs; this performs no new training or inference.

## Check Numerical Reproduction

After a full 929-observation evaluation, compare the actual report to each reference:

```bash
python scripts/verify_paper_results.py --report results/reproduced-validation/report.json --reference saved
python scripts/verify_paper_results.py --report results/reproduced-validation/report.json --reference paper
```

The comparison checks the cohort, resolution, seed, threshold, metric validity,
and all five primary values with absolute tolerance 0.00001. A mismatch returns
exit code 1 with actual, expected, and difference values. It never substitutes
paper values for measurements. `paper` uses the author's external main-table row;
reproducing that row requires the corresponding external model and protocol.

On 2026-09-20, full inference and evaluation were rerun on all 929 observations
at 4,096 points. All five primary metrics exactly reproduced the saved means.
The external main-table comparison matched CD-L1, CD-L2, and Dimension MAE but
did not match either F-score. Machine-readable checks are in
[reproduction_saved.json](reproduction_saved.json) and
[reproduction_paper.json](reproduction_paper.json).

`python scripts/render_paper_results.py` renders the same manuscript CSV values
into the English/Korean READMEs, benchmark page, and Hugging Face model-card template.
