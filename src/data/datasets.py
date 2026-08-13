"""Read-only manifest-backed dataset access for Stage 1 inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class ManifestPairDataset(Dataset[dict[str, str]]):
    """Expose validated manifest pairs without loading or transforming volumes."""

    def __init__(self, manifest_path: str | Path) -> None:
        report: dict[str, Any] = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if report.get("inspection_only") is not True:
            raise ValueError("Expected a Stage 1 inspection manifest.")
        self._pairs: list[dict[str, str]] = report.get("proposed_pairs", [])

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, str]:
        return dict(self._pairs[index])
