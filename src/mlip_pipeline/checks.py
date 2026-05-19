from __future__ import annotations

from pathlib import Path

from mlip_pipeline.utils.logging import info, warn


class TrainingSetMismatch(RuntimeError):
    """Raised when the training cfg count does not match expectations."""


def _count_cfgs(path: Path) -> int:
    """Count BEGIN_CFG markers in a .cfg file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "BEGIN_CFG":
                count += 1
    return count


def check_training_cfg_count(
    current_train_cfg: Path,
    prev_train_cfg: Path | None,
    prev_selected_cfg: Path | None,
    *,
    generation: int,
    strict: bool = True,
) -> None:
    """
    Verify that the training set grew by exactly the number of selected
    structures from the previous generation.

    Parameters
    ----------
    current_train_cfg   Path to this generation's train.cfg.
    prev_train_cfg      Path to the previous generation's merged train.cfg
                        (from state.merged_cfg of gen N-1). None for gen 0.
    prev_selected_cfg   Path to the previous generation's selected.cfg.
                        None if not available.
    generation          Current generation number (for log messages).
    strict              If True, raise TrainingSetMismatch on mismatch.
                        If False, only warn.
    """
    if not current_train_cfg.exists():
        raise FileNotFoundError(f"train.cfg not found: {current_train_cfg}")

    n_current = _count_cfgs(current_train_cfg)

    # First generation or no previous state: just report the count
    if prev_train_cfg is None or not Path(prev_train_cfg).exists():
        info(f"  [check] gen_{generation:02d} train.cfg: {n_current} cfg(s) (no previous gen to compare)")
        return

    n_prev_train = _count_cfgs(Path(prev_train_cfg))

    if prev_selected_cfg is None or not Path(prev_selected_cfg).exists():
        warn(
            f"  [check] gen_{generation:02d}: previous selected.cfg not found — "
            f"cannot verify count. Current: {n_current}, previous train: {n_prev_train}."
        )
        return

    n_selected = _count_cfgs(Path(prev_selected_cfg))
    expected   = n_prev_train + n_selected

    if n_current == expected:
        info(
            f"  [check] ✓ gen_{generation:02d} train.cfg: "
            f"{n_prev_train} (prev) + {n_selected} (selected) = {n_current} cfg(s)"
        )
    else:
        msg = (
            f"Training set size mismatch for gen_{generation:02d}:\n"
            f"  Expected : {n_prev_train} (prev train) + {n_selected} (selected) = {expected}\n"
            f"  Actual   : {n_current}\n"
            f"  Diff     : {n_current - expected:+d}\n"
            f"  train.cfg: {current_train_cfg}\n"
            f"  prev cfg : {prev_train_cfg}\n"
            f"  selected : {prev_selected_cfg}"
        )
        if strict:
            raise TrainingSetMismatch(msg)
        else:
            warn(f"  [check] ✗ {msg}")
