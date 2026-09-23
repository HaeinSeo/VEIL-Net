from __future__ import annotations

from pathlib import Path

import numpy as np

from rapc_net.datasets.pairs import load_npz_pair


class CompletionPairDataset:
    def __init__(self, pairs_dir: Path) -> None:
        self.paths = sorted(pairs_dir.rglob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No completion pair .npz files found in {pairs_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        item = load_npz_pair(self.paths[index])
        if item["complete"] is None:
            raise ValueError(f"{self.paths[index]} does not contain complete target")
        return item
