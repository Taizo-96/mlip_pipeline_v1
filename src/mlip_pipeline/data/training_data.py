"""
data/training_data.py — DEPRECATED.

This module has been superseded by the standalone script::

    scripts/prepare_initial_training.py

The script handles the one-time conversion of DeePMD-format system directories
to MLIP-3 .cfg files and is not part of the active-learning loop.  The
[training] section has been removed from Pb_loop.yaml.

This file is kept temporarily to avoid breaking any code that still imports
from this location.  It will be removed in a future cleanup.
"""
from __future__ import annotations

from pathlib import Path

from mlip_pipeline.models import PrepareTrainResult


def prepare_training_cfgs(config: dict, resolved_paths: dict) -> PrepareTrainResult:
    """Deprecated wrapper — delegates to the standalone script logic.

    Kept for backward compatibility only.  Prefer calling
    ``scripts/prepare_initial_training.py`` directly.
    """
    import warnings
    warnings.warn(
        "prepare_training_cfgs() is deprecated.  Use scripts/prepare_initial_training.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from scripts.prepare_initial_training import prepare  # type: ignore
    train_cfg = config.get("training", {})
    merged = prepare(
        data_root    = resolved_paths["data_root"],
        glob_pattern = train_cfg.get("input_glob", "Pb*"),
        output_dir   = resolved_paths["datasets_root"] / train_cfg.get("output_subdir", "pb_cfg"),
        merge_name   = train_cfg.get("merge_name", "train.cfg"),
    )
    merged_path = Path(merged) if not isinstance(merged, Path) else merged
    output_dir  = merged_path.parent if merged_path.is_file() else merged_path
    return PrepareTrainResult(
        output_dir   = output_dir,
        merged_cfg   = merged_path if merged_path.is_file() else None,
        generated_cfgs = [],
    )
