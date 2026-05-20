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
    Verify that the training set grew from the previous generation.

    The pipeline order is:  convert (gen N-1)  →  fit (gen N).
    ``convert`` appends the selected blocks from gen N-1 into train.cfg,
    so by the time fit runs for gen N, ``current_train_cfg`` already
    contains those blocks.  ``prev_train_cfg`` is the path recorded in
    ``state.merged_cfg`` after gen N-1's convert finished — i.e. the
    same file, at the same path, but the block count *at that moment*
    is what we stored.  Because both point at the same accumulating
    file the correct invariant is simply:

        current_blocks  ==  prev_blocks + n_selected

    where ``prev_blocks`` is re-counted from disk right now (they are
    the same file, but we recount to be safe) and ``n_selected`` comes
    from the previous generation's selected.cfg.

    Parameters
    ----------
    current_train_cfg   Path to this generation's train.cfg.
    prev_train_cfg      Path recorded as merged_cfg by gen N-1's convert
                        step.  None for gen 0 / first generation.
    prev_selected_cfg   Path to gen N-1's selected.cfg.  Used only for
                        the diagnostic diff message, not to compute
                        expected count.
    generation          Current generation number (for log messages).
    strict              If True, raise TrainingSetMismatch on mismatch.
                        If False, only warn.
    """
    if not current_train_cfg.exists():
        raise FileNotFoundError(f"train.cfg not found: {current_train_cfg}")

    n_current = _count_cfgs(current_train_cfg)

    # First generation or no previous state: just report the count
    if prev_train_cfg is None or not Path(prev_train_cfg).exists():
        info(
            f"  [check] gen_{generation:02d} train.cfg: "
            f"{n_current} cfg(s) (no previous gen to compare)"
        )
        return

    n_prev_train = _count_cfgs(Path(prev_train_cfg))

    # Selected count is informational only — used in the diff message.
    n_selected: int | None = None
    if prev_selected_cfg is not None and Path(prev_selected_cfg).exists():
        n_selected = _count_cfgs(Path(prev_selected_cfg))

    # The only meaningful invariant: current must be strictly larger than prev.
    # The exact delta should equal n_selected, but we only hard-fail if the
    # file did not grow at all (append never happened).
    delta = n_current - n_prev_train

    if n_selected is not None and delta == n_selected:
        info(
            f"  [check] ✓ gen_{generation:02d} train.cfg: "
            f"{n_prev_train} (prev) + {n_selected} (selected) = {n_current} cfg(s)"
        )
        return

    if delta > 0:
        # Grew, but not by exactly n_selected — warn, don't fail
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
        f"  prev train : {n_prev_train} blocks\n"
        f"  selected   : {sel_str} blocks\n"
        f"  expected   : {expected}\n"
        f"  actual     : {n_current}  (delta {delta:+d})\n"
        f"  train.cfg  : {current_train_cfg}\n"
        f"  prev cfg   : {prev_train_cfg}\n"
        f"  selected   : {prev_selected_cfg}\n"
        f"\nThe convert step for gen_{generation-1:02d} likely did not run or "
        f"appended to the wrong file."
    )
    if strict:
        raise TrainingSetMismatch(msg)
    else:
        warn(f"  [check] ✗ {msg}")
