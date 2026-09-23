# Research results

[Back to the project](../README.md#results)

The main README reports the measured reference checkpoint. This page preserves
the complete manuscript tables and distinguishes the runs behind them. Machine-
readable values and metric definitions are in [benchmarks](../benchmarks/README.md).

## Complete tables

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

## Reproduction record

The reference checkpoint was re-evaluated over all 929 observations on
2026-09-23, using seed 0 and 4,096 evaluation points. Its five primary metrics
matched the previous saved evaluation exactly.

- [Full evaluation report](../benchmarks/bundled_evaluation.json)
- [Per-observation measurements](../benchmarks/bundled_samples.csv)
- [Saved-reference comparison](../benchmarks/bundled_saved_verdict.json)
- [External-target comparison](../benchmarks/bundled_paper_verdict.json)
- [Frozen validation protocol](../benchmarks/validation_protocol.json)

The external-target comparison fails on F@0.03 and F@0.05. The target search
retains this distinction and selects a winner only when all five targets pass
under the same evaluation protocol.

The qualitative figures in the README are author-supplied illustrations. Their
source images do not identify checkpoint hashes or sample IDs, so they are not
used as evidence that the reference checkpoint reproduces the external run.
