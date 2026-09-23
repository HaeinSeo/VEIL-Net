# Contributing

Start with the public API in `veil_net/`. Changes to the checkpoint-compatible
implementation in `rapc_net/` should preserve state-dict names or document an
explicit migration.

```bash
python -m pip install -e ".[dev,hub,research]"
python -m veil_net smoke
python -m pytest -q
python -m ruff check
```

Include a small regression case for changes to inference, metrics, serialization,
or training behavior. Report model hashes, sampling seeds, sample IDs, and metric
resolution when discussing benchmark changes. Keep measured results and software
smoke checks distinct. Use a new output directory for each evaluation.

Do not commit datasets, model weights, access tokens, local environment folders,
or generated prediction archives. The release script stages an explicit source
allowlist; weights are distributed as a separate model artifact.
