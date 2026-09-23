---
library_name: pytorch
tags:
- point-cloud-completion
- 3d
- geometry
- veil-net
language:
- en
license: mit
---

# VEIL-Net

**Visible-to-Entire Surface Inference via Local Geometry for Occluded 3D Objects**

VEIL-Net infers a complete object surface from a partial XYZ point cloud.
It combines local geometric descriptors, coordinate-conditioned surface queries,
two-stage folding, and observed-surface integration. The standard configuration
maps 2,048 observed points to 16,384 surface points using native PyTorch operators.

## Research Results

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

## Exported Checkpoint Reference

On the saved 929-observation validation cohort, the coverage-balance checkpoint
reduces **CD-L2 by 16.41%** and **Dimension MAE by 33.27%** against task-finetuned
SnowflakeNet.

| Model | CD-L1 ↓ | CD-L2 ↓ | F@0.03 ↑ | F@0.05 ↑ | Dim. MAE ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| SnowflakeNet FT | 0.42271 | 8.34760 | 0.65293 | **0.75463** | 0.34238 |
| VEIL-Net reference checkpoint | **0.38629** | **6.97737** | **0.65844** | 0.75428 | **0.22846** |

These values use 4,096-point metric sampling with seed 0 and apply to source
checkpoint SHA-256 `1e7cac6daed3f5ffab7a187fe5dd191d549de0f95525d7e870a4a0d5715a215d`.
Compare this identity with the export provenance below. The full benchmark
record also reports missing-region and observed-region errors.

## Usage

Install the VEIL-Net code repository with `pip install ".[hub]"`, then:

```python
from pathlib import Path
import numpy as np
from veil_net import VEILNet
from veil_net.inference import predict

model = VEILNet.from_pretrained(Path("./veil-model"), device="cpu")
result = predict(model, np.load("partial.npy", allow_pickle=False))
np.save("complete.npy", result["prediction_complete"])
```

After publication, the local directory can be replaced by the actual Hub model
ID. Pin `revision` to a commit for reproducible remote downloads.

Input is a finite, nonempty `[N, 3]` array for one segmented object. Sampling to
the configured point count is deterministic. Output uses the input coordinate
frame and units; GT is not needed at inference. Extra channels and entire scene
clouds are not part of this interface.

## Architecture and Training

The geometry-query configuration uses 128 spatial anchors, up to 16 neighbors
per anchor, and 256 predicted patch centers. Training uses paired partial and
complete object surfaces. Internal `risk` names are retained for compatibility;
the signal represents learned observation quality.

## Evaluation

Benchmark numbers must be associated with the checkpoint hash and evaluation
protocol that produced them. See the code repository's `benchmarks/README.md`
for the saved 929-observation validation comparison and separately reported
external results. Exporting another checkpoint does not inherit those scores.

This model targets object-level surface completion. A quick software smoke test
checks execution and serialization; benchmark evaluation requires the prepared
partial/complete pairs and a fixed sampling protocol.

## Files

- `model.safetensors`: inference weights without optimizer state.
- `config.json`: complete architecture configuration.
- `provenance.json`: source and exported-file hashes, checkpoint epoch, and version.

RAPC-Net is the historical code/checkpoint name of VEIL-Net. The public class
preserves parameter names, allowing strict loading of the original checkpoints.
