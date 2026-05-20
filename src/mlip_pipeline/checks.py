from __future__ import annotations

import json
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


def _prev_block_count_from_manifest(prev_convert_dir: Path | None) -> int | None:
    """
    Load the total_block_count recorded by convert for gen N-1.

    Returns None if the manifest does not exist (old runs / first gen).
    This is the only safe way to get the pre-fit block count when
    prev_train_cfg and current_train_cfg are the same accumulating file
    -- recounting from disk would give the same number twice.
    """
    if prev_convert_dir is None:
        return None
    manifest = Path(prev_convert_dir) / "convert_manifest.json"
    if not manifest.exists():
        return None
    try:
        d = json.loads(manifest.read_text(encoding="utf-8"))
        val = d.get("total_block_count")
        return int(val) if val is not None else None
    except Exception:
        return None


def check_training_cfg_count(
    current_train_cfg: Path,
    prev_train_cfg: Path | None,
    prev_selected_cfg: Path | None,
    *,
    generation: int,
    strict: bool = True,
    prev_convert_dir: Path | None = None,
) -> None:
    """
    Verify that the training set grew by the expected number of blocks.

    The pipeline accumulates all generations into a single train.cfg, so
    prev_train_cfg and current_train_cfg often point at the same file.
    Recounting from disk would always give delta=0.  Instead we read
    the block count that was recorded *at convert time* from the
    convert_manifest.json written by ConvertResult.save_manifest().

    Parameters
    ----------
    current_train_cfg   Path to this generation's train.cfg (read live).
    prev_train_cfg      Path to the previous generation's merged train.cfg.
                        Used only for the fallback disk-recount path and
                        the diagnostic message.
    prev_selected_cfg   Path to gen N-1's selected.cfg.  Informational.
    generation          Current generation number.
    strict              Raise TrainingSetMismatch on mismatch if True.
    prev_convert_dir    Directory that holds gen N-1's convert_manifest.json
                        (i.e. datasets/converted_cfg/gen_13/).  When supplied
                        the manifest's total_block_count is used as the
                        authoritative prev baseline instead of a disk recount.
    """
    if not current_train_cfg.exists():
        raise FileNotFoundError(f"train.cfg not found: {current_train_cfg}")

    n_current = _count_cfgs(current_train_cfg)

    # First generation or no previous state
    if prev_train_cfg is None and prev_convert_dir is None:
        info(
            f"  [check] gen_{generation:02d} train.cfg: "
            f"{n_current} cfg(s) (no previous gen to compare)"
        )
        return

    # Preferred: read the count that was recorded at convert-time
    n_prev_train = _prev_block_count_from_manifest(prev_convert_dir)

    if n_prev_train is None:
        # Fallback: recount from disk (only correct when files differ)
        if prev_train_cfg is None or not Path(prev_train_cfg).exists():
            info(
                f"  [check] gen_{generation:02d} train.cfg: "
                f"{n_current} cfg(s) (no previous gen to compare)"
            )
            return
        n_prev_train = _count_cfgs(Path(prev_train_cfg))
        warn(
            f"  [check] gen_{generation:02d}: convert_manifest.json not found — "
            f"falling back to disk recount of prev_train_cfg "
            f"({n_prev_train} blocks).  This may be inaccurate if both paths "
            f"point at the same accumulating file."
        )

    # Selected count is informational only
    n_selected: int | None = None
    if prev_selected_cfg is not None and Path(prev_selected_cfg).exists():
        n_selected = _count_cfgs(Path(prev_selected_cfg))

    delta = n_current - n_prev_train

    if n_selected is not None and delta == n_selected:
        info(
            f"  [check] ✓ gen_{generation:02d} train.cfg: "
            f"{n_prev_train} (prev) + {n_selected} (selected) = {n_current} cfg(s)"
        )
        return

    if delta > 0:
        sel_str = str(n_selected) if n_selected is not None else "unknown"
        warn(
            f"  [check] gen_{generation:02d} train.cfg grew by {delta} "
            f"(expected {sel_str} from selected.cfg) — "
            f"proceeding but counts do not match exactly.\n"
            f"  current: {n_current}  prev: {n_prev_train}  "
            f"train.cfg: {current_train_cfg}"
        )
        return

    # delta <= 0: the file did not grow — convert never appended
    sel_str  = str(n_selected) if n_selected is not None else "unknown"
    expected = n_prev_train + (n_selected or 0)
    msg = (
        f"Training set did not grow for gen_{generation:02d}:\n"
        f"  prev train : {n_prev_train} blocks  "
        f"(from convert_manifest of gen_{generation - 1:02d})\n"
        f"  selected   : {sel_str} blocks\n"
        f"  expected   : {expected}\n"
        f"  actual     : {n_current}  (delta {delta:+d})\n"
        f"  train.cfg  : {current_train_cfg}\n"
        f"  prev cfg   : {prev_train_cfg}\n"
        f"  selected   : {prev_selected_cfg}\n"
        f"\nThe convert step for gen_{generation - 1:02d} likely did not run or "
        f"appended to the wrong file."
    )
    if strict:
        raise TrainingSetMismatch(msg)
    else:
        warn(f"  [check] ✗ {msg}")
