# src/mlip_pipeline/models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class PrepareTrainResult:
    output_dir: Path
    merged_cfg: Path | None
    generated_cfgs: Sequence[Path]


@dataclass
class FitResult:
    run_dir: Path
    model_path: Path
    log_path: Path


@dataclass
class SelectionResult:
    select_root: Path
    manifest_path: Path
    selected_cfg_paths: list[Path]
    selected_count: int
