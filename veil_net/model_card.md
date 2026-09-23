---
library_name: pytorch
license: mit
tags:
  - point-cloud-completion
  - 3d-vision
  - veil-net
---

# VEIL-Net

Visible-to-Entire Surface Inference via Local Geometry for Occluded 3D Objects.

## Input and output

The model completes a segmented object's partial XYZ point cloud. The default
configuration takes 2,048 points and produces 16,384 points in the input frame
and units. RGB, ground-truth surfaces, and object dimensions are not inference
inputs. Configuration is stored in `config.json`.

## Usage

Install the [implementation](https://github.com/HaeinSeo/VEIL-Net), then load this
export directory:

```python
import numpy as np

from veil_net import VEILNet
from veil_net.inference import predict

model = VEILNet.from_pretrained("model")
complete = predict(model, np.load("partial.npy", allow_pickle=False))["prediction_complete"]
```

## Evaluation

This card is generated during checkpoint export. Publish the checkpoint's own
evaluation report alongside its weights; reference scores from the source
repository do not automatically apply to a newly trained checkpoint.
The appended provenance records the exported configuration and source identity.
