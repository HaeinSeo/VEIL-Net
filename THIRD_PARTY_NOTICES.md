# Third-Party Notices

The MIT license at the repository root applies to original VEIL-Net project code.
Dependencies such as PyTorch, NumPy, PyYAML, safetensors, and huggingface_hub retain
their respective licenses. They are installed separately and are not vendored.

The `adapters/` directory contains integration code for external baselines.
PoinTr/AdaPoinTr, SnowflakeNet, GRNet, and 3DAttriFlow implementations and their
weights must be obtained from their respective authors under the original terms.
The curated release does not include their repositories or checkpoints.

T-LESS, TUD-L, YCB-Video, and other source datasets/CAD assets retain their original
licenses. Prepared point clouds and RGB-D images are not included. The compact
benchmark files contain aggregate evaluation values and sample identifiers.

Before redistributing any additional third-party code, weights, images, or data,
include the corresponding attribution and license alongside that artifact.
