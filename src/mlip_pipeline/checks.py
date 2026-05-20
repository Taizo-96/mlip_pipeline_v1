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
    Load the prev_block_count recorded by gen N-1's convert step.

    prev_block_count is the number of BEGIN_CFG blocks in train.cfg
    BEFORE gen N-1's convert appended its new structures — i.e. the
    count that gen N-1's fit actually trained on.

    This is the correct baseline for verifying gen N's training set:
        current_blocks - prev_block_count == n_selected_by_gen_N-1

    Using total_block_count (post-convert) would give delta=0 because
    current_train_cfg and the accumulating train.cfg are the same file.

    Returns None if the manifest does not exist (old runs / first gen).
    """
    if prev_convert_dir is None:
        return None
    manifest = Path(prev_convert_dir) / "convert_manifest.json"
    if not manifest.exists():
        return None
    try:
        d = json.loads(manifest.read_text(encoding="utf-8"))
        val = d.get("prev_block_count")
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
    Verify that train.cfg grew by exactly the number of selected blocks.

    The pipeline accumulates all generations into a single train.cfg, so
    prev_train_cfg and current_train_cfg point at the same file.
    Recounting from disk always gives delta=0.

    Instead, read prev_block_count from gen N-1's convert_manifest.json —
    that is the block count at the moment BEFORE gen N-1's convert ran,
    i.e. the count gen N-1 fit actually trained on.  The invariant is:

        current_blocks - prev_block_count  ==  n_selected (gen N-1)

    Parameters
    ----------
    current_train_cfg   Path to train.cfg (read live from disk).
    prev_train_cfg      Path recorded as merged_cfg by gen N-1 — used
                        only for the diagnostic message and disk-recount
                        fallback.
    prev_selected_cfg   Path to gen N-1's selected.cfg (informational).
    generation          Current generation number.
    strict              Raise TrainingSetMismatch on mismatch if True.
    prev_convert_dir    Directory containing gen N-1's
                        convert_manifest.json (e.g.
                        datasets/converted_cfg/gen_13/).  Required for
                        the manifest-based baseline; falls back to disk
                        recount if absent.
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

    # Preferred: read prev_block_count stamped at convert-time
    # (= what gen N-1 fit trained on, BEFORE the append)
    n_prev_train = _prev_block_count_from_manifest(prev_convert_dir)

    if n_prev_train is None:
        # Fallback: disk recount — only accurate when the files differ
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
            f"{n_prev_train} (prev fit) + {n_selected} (selected) = {n_current} cfg(s)"
        )
        return

    if delta > 0:
        sel_str = str(n_selected) if n_selected is not None else "unknown"
        warn(
            f"  [check] gen_{generation:02d} train.cfg grew by {delta} "
            f"(expected {sel_str} from selected.cfg) — "
            f"proceeding but counts do not match exactly.\n"
            f"  current: {n_current}  prev fit: {n_prev_train}  "
            f"train.cfg: {current_train_cfg}"
        )
        return

    # delta <= 0: file did not grow — convert never appended
    sel_str  = str(n_selected) if n_selected is not None else "unknown"
    expected = n_prev_train + (n_selected or 0)
    msg = (
        f"Training set did not grow for gen_{generation:02d}:\n"
        f"  prev fit   : {n_prev_train} blocks  "
        f"(prev_block_count from gen_{generation - 1:02d} convert_manifest)\n"
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
