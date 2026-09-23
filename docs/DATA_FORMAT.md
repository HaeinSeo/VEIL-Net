# Data Contract

## Single-Object Inference

Use a NumPy `.npy` array with shape `[N, 3]`, or an `.npz` with a `partial` array.
Coordinates must be finite. Input must contain at least one point and represent
one already segmented object. The public inference helper samples to the model's
configured input count, using replacement only when the cloud is too small.
An already prepared input with the correct count keeps its original point order.

The network performs its own input-only centering/scaling and restores output
coordinates to the input frame. Use the same frame and units for prediction,
target, and observations when computing metrics. Thresholds such as 0.03 refer
to those coordinate units.

```text
partial.npy                   float32 [N, 3]
completion.npz
  input_partial               float32 [2048, 3]
  prediction_complete         float32 [16384, 3]
  generated_points            float32 [16384, 3]
  coarse_points               float32 [256, 3]
  point_risk                  float32 [2048]
```

Additional diagnostic arrays can be present. They do not change the surface
output contract. Output dimensions follow the saved model configuration.

## Training and Evaluation Pairs

```text
pairs/<split>/<dataset>/<sample_id>.npz
  partial                     float32 [2048, 3]
  complete                    float32 [16384, 3]
  risk_target                 float32 [2048] (training supervision, when provided)
```

Each sample ID must be unique across the selected split. Complete targets must
be CAD-aligned or otherwise registered to the observed cloud in the prepared
pair frame. The original pair-generation code is in `rapc_net/datasets/pairs.py`;
RGB-D construction uses depth, object mask, camera intrinsics, and model pose.
Datasets and CAD meshes are obtained under their original licenses.

For evaluation, `--protocol` selects exactly its `sample_ids`; missing or
duplicate IDs raise an error. `--max-samples` selects a seeded subset from that
cohort. All selected IDs, input file hashes, and actual metric resolution are
saved. Prepared evaluation inputs must already have the configured point count.

The provided validation protocol excludes overlapping training observation IDs.
For a new dataset, prepare train/validation separation before training and use
a protocol describing that experiment's intended generalization setting.
